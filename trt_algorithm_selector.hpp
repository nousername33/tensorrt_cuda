#pragma once
// Tactics cache：基于 nvinfer1::IAlgorithmSelector 固定每层选中的 tactic，
// 解决同一模型多次 porting 因 tactic 选择不确定导致的推理精度不一致问题。
// 与 timing cache（ITimingCache，只加速计时）完全正交，互不干扰。
//
// 【换名免疫设计】name 仅作 "本次 build 内" 把 IAlgorithmContext 定位到当前网络
// ILayer 的临时 key（直查，不做任何复合名解析 / 格式假设）；持久化 cache 的 key
// 完全由结构生成，不含、不比较、不匹配 layerName。因此层名随机改动、整体置换、
// 甚至一个名字被赋给结构不同的层，都不会让相同结构的层选错 tactic。
//   命中 ILayer（非融合层）→ 富结构 key（含
//   layerType/precision/config/dtype）。 未命中（融合层 / TRT 内部层 /
//   重名层）→ 名字无关的 OPAQUE::shape 兜底 key， 并用 " 哪条记录的 (impl,tactic)
//   出现在本层候选里 " 做消歧，让每层仍选回自己的 tactic。
//
// 限定编译范围为 [8000, 10000)，范围外不参与编译。
#if TRT_VERSION_FOR_MINFER >= 8000 && TRT_VERSION_FOR_MINFER < 10000

#include "trt_utils.hpp"

#include <cctype>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace minfer {

//cache key / 格式 版本；升级后旧 cache 会被 load() 判为过期而忽略（触发重新
//record）。1.7：去除层名解析，key 经直查获取结构；每个 key 存多条记录。
static constexpr char kTacticsCacheVersion[] = "1.7";

// TRT >= 8200 才有 IAlgorithm::getAlgorithmIOInfoByIndex（IO format 匹配）。
#if TRT_VERSION_FOR_MINFER >= 8200
#define TACS_HAS_IOINFO 1
#else
#define TACS_HAS_IOINFO 0
#endif

//---------- dtype /dims 转字符串，用于拼 key ----------

static inline std::string tacs_dtype_to_str(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return "FP32";
    case nvinfer1::DataType::kHALF:
      return "FP16";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kINT32:
      return "INT32";
    case nvinfer1::DataType::kBOOL:
      return "BOOL";
#if TRT_VERSION_FOR_MINFER >= 8500
    case nvinfer1::DataType::kUINT8:
      return "UINT8";
#endif  // TRT_VERSION_FOR_MINFER >= 8500
#if TRT_VERSION_FOR_MINFER >= 8600
    case nvinfer1::DataType::kFP8:
      return "FP8";
#endif  // TRT_VERSION_FOR_MINFER >= 8600
    default:
      return "UNKNOWN";
  }
}

static inline std::string tacs_dims_to_str(nvinfer1::Dims const &dims) {
  std::ostringstream oss;
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    if (i > 0) {
      oss << "x";
    }
    oss << dims.d[i];
  }
  return oss.str();
}

//---------- IO format（algorithm 派生：layout /dtype/strides / 向量化）
//---------- 只能从 IAlgorithm（选中的 choice）拿到，属 "算法身份"，不进查找
//key， 存进记录 value，供 selectAlgorithms 在同 (impl,tactic)
// 的多个候选间做精确 tie-break。
struct IOFormat {
  int32_t format{0};              // nvinfer1::TensorFormat（layout）
  int32_t dtype{0};               // nvinfer1::DataType
  std::vector<int32_t> strides;   // Dims
  int64_t vec_dim{-1};            // 向量化维
  int64_t comps{1};               // 每元素分量数
};

//---------- layer name -> ILayer*，build 前从 network 遍历一次构建 ----------
// 仅用于 "本次 build 内" 把 context 名字直查回当前 ILayer（取类型 / 结构属性 / 网络
// 声明 dtype）。name 不进 key、不跨 build 复用。
using TacsLayerMap = std::unordered_map<std::string, nvinfer1::ILayer *>;

static inline TacsLayerMap
tacs_build_layer_map(nvinfer1::INetworkDefinition const &network) {
  TacsLayerMap layer_map;
  std::unordered_map<std::string, int32_t> name_count;
  int32_t nb_layers = network.getNbLayers();
  for (int32_t i = 0; i < nb_layers; ++i) {
    nvinfer1::ILayer *layer = network.getLayer(i);
    if (layer != nullptr && layer->getName() != nullptr) {
      std::string name = layer->getName();
      layer_map[name] = layer;
      name_count[name]++;
    }
  }
  // 重名剔除：同一 build 内出现 >1
  // 次的名字视为歧义，从表中移除，使这些层直查判定为 "未命中"→ 退到 OPAQUE+shape
  // 兜底（名字无关），而不是被解析 / 覆盖成错误的层。
  for (auto const &kv : name_count) {
    if (kv.second > 1) {
      layer_map.erase(kv.first);
    }
  }
  return layer_map;
}

// 只对 tactic 差异最大、最容易在同 shape 下发生碰撞的几类算子补充 extraAttrs，
// 其它层类型保持纯 type+shape 粒度，不做通用反射设计。全部取自结构属性。
static inline std::string tacs_extra_attrs(nvinfer1::ILayer const *layer) {
  if (layer == nullptr) {
    return "";
  }
  std::ostringstream oss;
  switch (layer->getType()) {
    case nvinfer1::LayerType::kCONVOLUTION: {
      auto const *conv = static_cast<nvinfer1::IConvolutionLayer const *>(layer);
      //k=kernel、nom = 输出通道数 是 conv tactic 的关键区分维度（此前缺失，导致同
      //in/out shape 但不同 kernel 的 conv 碰撞成同一 key）。
      oss << "::k" << tacs_dims_to_str(conv->getKernelSizeNd()) << "nom"
          << conv->getNbOutputMaps() << "s"
          << tacs_dims_to_str(conv->getStrideNd()) << "p"
          << tacs_dims_to_str(conv->getPaddingNd()) << "d"
          << tacs_dims_to_str(conv->getDilationNd()) << "g"
          << conv->getNbGroups();
      break;
    }
    case nvinfer1::LayerType::kDECONVOLUTION: {
      auto const *deconv =
          static_cast<nvinfer1::IDeconvolutionLayer const *>(layer);
      oss << "::k" << tacs_dims_to_str(deconv->getKernelSizeNd()) << "nom"
          << deconv->getNbOutputMaps() << "s"
          << tacs_dims_to_str(deconv->getStrideNd()) << "p"
          << tacs_dims_to_str(deconv->getPaddingNd()) << "g"
          << deconv->getNbGroups();
      break;
    }
    case nvinfer1::LayerType::kMATRIX_MULTIPLY: {
      auto const *mm = static_cast<nvinfer1::IMatrixMultiplyLayer const *>(layer);
      oss << "::op" << static_cast<int32_t>(mm->getOperation(0))
          << static_cast<int32_t>(mm->getOperation(1));
      break;
    }
    case nvinfer1::LayerType::kPOOLING: {
      auto const *pool = static_cast<nvinfer1::IPoolingLayer const *>(layer);
      oss << "::t" << static_cast<int32_t>(pool->getPoolingType()) << "w"
          << tacs_dims_to_str(pool->getWindowSizeNd()) << "s"
          << tacs_dims_to_str(pool->getStrideNd()) << "p"
          << tacs_dims_to_str(pool->getPaddingNd());
      break;
    }
    case nvinfer1::LayerType::kELEMENTWISE: {
      auto const *ew = static_cast<nvinfer1::IElementWiseLayer const *>(layer);
      oss << "::op" << static_cast<int32_t>(ew->getOperation());
      break;
    }
    case nvinfer1::LayerType::kREDUCE: {
      auto const *red = static_cast<nvinfer1::IReduceLayer const *>(layer);
      oss << "::op" << static_cast<int32_t>(red->getOperation()) << "ax"
          << red->getReduceAxes() << "k" << (red->getKeepDimensions() ? 1 : 0);
      break;
    }
    case nvinfer1::LayerType::kACTIVATION: {
      auto const *act = static_cast<nvinfer1::IActivationLayer const *>(layer);
      oss << "::t" << static_cast<int32_t>(act->getActivationType());
      break;
    }
    case nvinfer1::LayerType::kUNARY: {
      auto const *un = static_cast<nvinfer1::IUnaryLayer const *>(layer);
      oss << "::op" << static_cast<int32_t>(un->getOperation());
      break;
    }
    case nvinfer1::LayerType::kPLUGIN_V2: {
      //plugin 用 plugin 类型名 + 版本（类标识，与实例层名无关 →
      // 换名免疫）区分不同 plugin，避免同 io shape 的不同 plugin 碰撞。
      auto *pl = const_cast<nvinfer1::IPluginV2Layer *>(
          static_cast<nvinfer1::IPluginV2Layer const *>(layer));
      nvinfer1::IPluginV2 &p = pl->getPlugin();
      char const *ptype = p.getPluginType();
      char const *pver = p.getPluginVersion();
      oss << "::plg" << (ptype != nullptr ? ptype : "") << "/"
          << (pver != nullptr ? pver : "");
      break;
    }
    default:
      break;
  }
  return oss.str();
}

//layerType 结构枚举字符串：
//layer->getType()（结构属性），对完全换名免疫。只映射常见类型，其余用枚举数值（同
// TRT 版本内稳定）。
static inline std::string tacs_layer_type_str(nvinfer1::LayerType t) {
  switch (t) {
    case nvinfer1::LayerType::kCONVOLUTION:
      return "CONV";
    case nvinfer1::LayerType::kDECONVOLUTION:
      return "DECONV";
    case nvinfer1::LayerType::kMATRIX_MULTIPLY:
      return "MATMUL";
    case nvinfer1::LayerType::kPOOLING:
      return "POOL";
    case nvinfer1::LayerType::kACTIVATION:
      return "ACT";
    case nvinfer1::LayerType::kELEMENTWISE:
      return "ELTWISE";
    case nvinfer1::LayerType::kSCALE:
      return "SCALE";
    case nvinfer1::LayerType::kSOFTMAX:
      return "SOFTMAX";
    case nvinfer1::LayerType::kSHUFFLE:
      return "SHUFFLE";
    case nvinfer1::LayerType::kSLICE:
      return "SLICE";
    case nvinfer1::LayerType::kRESIZE:
      return "RESIZE";
    case nvinfer1::LayerType::kCONCATENATION:
      return "CONCAT";
    case nvinfer1::LayerType::kUNARY:
      return "UNARY";
    case nvinfer1::LayerType::kREDUCE:
      return "REDUCE";
    case nvinfer1::LayerType::kGATHER:
      return "GATHER";
    case nvinfer1::LayerType::kPLUGIN_V2:
      return "PLUGIN";
    default:
      return "LT" + std::to_string(static_cast<int32_t>(t));
  }
}

// 仅直查：context 名字当作 "本次 build 内" 的唯一定位 key，命中即返回对应
// ILayer； 不做任何复合名解析 / 格式假设。融合层（复合名）、TRT 内部层、
// 被重名剔除的歧义层都会直查不到 → 返回 nullptr，交由 tacs_extract_key 走
// OPAQUE+shape 名字无关兜底。
static inline nvinfer1::ILayer const *
tacs_resolve_layer(std::string const &raw_name, TacsLayerMap const &layer_map) {
  if (raw_name.empty()) {
    return nullptr;
  }
  auto it = layer_map.find(raw_name);
  return (it != layer_map.end()) ? it->second : nullptr;
}

static inline bool tacs_dims_equal(nvinfer1::Dims const &a,
                                   nvinfer1::Dims const &b) {
  if (a.nbDims != b.nbDims) {
    return false;
  }
  for (int32_t i = 0; i < a.nbDims; ++i) {
    if (a.d[i] != b.d[i]) {
      return false;
    }
  }
  return true;
}

//shape 字段：默认用 kOPT；仅当 min/max 与 opt 不同（动态 shape / 多
//profile）时才 展开成 min|opt|max，避免不同 profile 的同 opt
// 层被错误折叠到同一 key。静态 shape 下三者相等 → 只输出 opt，key
// 保持紧凑。全部来自 context，可复现、与名字无关。
static inline std::string
tacs_shape_field(nvinfer1::IAlgorithmContext const &ctx, int32_t idx) {
  nvinfer1::Dims opt = ctx.getDimensions(idx, nvinfer1::OptProfileSelector::kOPT);
  nvinfer1::Dims mn = ctx.getDimensions(idx, nvinfer1::OptProfileSelector::kMIN);
  nvinfer1::Dims mx = ctx.getDimensions(idx, nvinfer1::OptProfileSelector::kMAX);
  std::string so = tacs_dims_to_str(opt);
  if (tacs_dims_equal(mn, opt) && tacs_dims_equal(mx, opt)) {
    return so;
  }
  return tacs_dims_to_str(mn) + "|" + so + "|" + tacs_dims_to_str(mx);
}

// 提取 key。name 仅用于直查定位到本次网络的 ILayer（不进 key、不解析格式）：
//
//   命中 ILayer（非融合的单层）→ 富结构 key：
//     layerType::[prec=P]::in [dtype:shape]::out [dtype:shape][::config]
//   未命中（融合层 / TRT 内部层 / 被重名剔除的层）→ 名字无关兜底 key：
//     OPAQUE::in [dtype:shape]::out [dtype:shape]
//     （拿不到网络声明 dtype 时该项为 AUTO；靠 shape + 候选消歧区分）
//
//shape 来自 context 的 min/opt/max（与算法无关，天然稳定、换名免疫）；
//dtype 命中层时取网络声明 ITensor::getType()，否则 AUTO；config 为 conv
//kernel/nom/stride/... 及 plugin 类型 / 版本等结构属性。
//layout 属 algorithm 派生，不进 key（见 IOFormat），只用于选择阶段 tie-break。
static inline std::string
tacs_extract_key(nvinfer1::IAlgorithmContext const &context,
                 nvinfer1::IAlgorithm const &algo,
                 TacsLayerMap const &layer_map) {
  (void)algo;
  std::ostringstream key;
  std::string raw_name = context.getName();
  nvinfer1::ILayer const *layer = tacs_resolve_layer(raw_name, layer_map);

  if (layer != nullptr) {
    // 直查命中：非融合单层，类型签名 = 该层 getType()（结构枚举）。
    key << tacs_layer_type_str(layer->getType());
    //precision：层显式设置的计算精度（结构属性）。区分同 io dtype
    // 但计算精度不同的层。仅在显式 set 时拼入以避免噪声。
    if (layer->precisionIsSet()) {
      key << "::prec=" << tacs_dtype_to_str(layer->getPrecision());
    }
  } else {
    // 未命中：融合层 / TRT 内部层 / 被重名剔除的层。用固定前缀 OPAQUE + 后面的
    //in/out shape 区分（名字无关）；同 shape
    // 的不同结构由候选消歧兜底。固定前缀保证 OPAQUE 兜底 key 与富 key（CONV::
    // 等）永不碰撞。
    key << "OPAQUE";
  }

  int32_t layer_nb_inputs = (layer != nullptr) ? layer->getNbInputs() : 0;
  int32_t layer_nb_outputs = (layer != nullptr) ? layer->getNbOutputs() : 0;
  int32_t nb_inputs = context.getNbInputs();
  int32_t nb_outputs = context.getNbOutputs();

  // in[...]
  key << "::in[";
  for (int32_t i = 0; i < nb_inputs; ++i) {
    if (i > 0) {
      key << ",";
    }
    nvinfer1::DataType dtype;
    bool dtype_resolved = false;
    if (layer != nullptr && i < layer_nb_inputs) {
      nvinfer1::ITensor const *t = layer->getInput(i);
      if (t != nullptr) {
        dtype = t->getType();
        dtype_resolved = true;
      }
    }
    key << (dtype_resolved ? tacs_dtype_to_str(dtype) : "AUTO") << ":"
        << tacs_shape_field(context, i);
  }
  key << "]";

  // out[...]
  key << "::out[";
  for (int32_t i = 0; i < nb_outputs; ++i) {
    if (i > 0) {
      key << ",";
    }
    nvinfer1::DataType dtype;
    bool dtype_resolved = false;
    if (layer != nullptr && i < layer_nb_outputs) {
      nvinfer1::ITensor const *t = layer->getOutput(i);
      if (t != nullptr) {
        dtype = t->getType();
        dtype_resolved = true;
      }
    }
    key << (dtype_resolved ? tacs_dtype_to_str(dtype) : "AUTO") << ":"
        << tacs_shape_field(context, nb_inputs + i);
  }
  key << "]";

  //config（extraAttrs）：conv kernel/stride/pad/dilation/groups、plugin 类型等
  if (layer != nullptr) {
    key << tacs_extra_attrs(layer);
  }
  return key.str();
}

//---------- IO format 采集与匹配（选择阶段的精确 tie-break） ----------

#if TACS_HAS_IOINFO
static inline std::vector<IOFormat>
tacs_collect_ioinfo(nvinfer1::IAlgorithm const &algo, int32_t nb_io) {
  std::vector<IOFormat> out;
  for (int32_t j = 0; j < nb_io; ++j) {
    nvinfer1::IAlgorithmIOInfo const *info = algo.getAlgorithmIOInfoByIndex(j);
    if (info == nullptr) {
      continue;
    }
    IOFormat f;
    f.format = static_cast<int32_t>(info->getTensorFormat());
    f.dtype = static_cast<int32_t>(info->getDataType());
    nvinfer1::Dims s = info->getStrides();
    for (int32_t k = 0; k < s.nbDims; ++k) {
      f.strides.push_back(static_cast<int32_t>(s.d[k]));
    }
    f.vec_dim = info->getVectorizedDim();
    f.comps = info->getComponentsPerElement();
    out.push_back(std::move(f));
  }
  return out;
}

// 该 choice 的完整 IO format 是否与记录一致。
static inline bool tacs_ioinfo_matches(nvinfer1::IAlgorithm const &algo,
                                       std::vector<IOFormat> const &ref,
                                       int32_t nb_io) {
  if (static_cast<int32_t>(ref.size()) != nb_io) {
    return false;
  }
  for (int32_t j = 0; j < nb_io; ++j) {
    nvinfer1::IAlgorithmIOInfo const *info = algo.getAlgorithmIOInfoByIndex(j);
    if (info == nullptr) {
      return false;
    }
    IOFormat const &r = ref[j];
    if (static_cast<int32_t>(info->getTensorFormat()) != r.format ||
        static_cast<int32_t>(info->getDataType()) != r.dtype ||
        info->getVectorizedDim() != r.vec_dim ||
        info->getComponentsPerElement() != r.comps) {
      return false;
    }
    nvinfer1::Dims s = info->getStrides();
    if (static_cast<int32_t>(r.strides.size()) != s.nbDims) {
      return false;
    }
    for (int32_t k = 0; k < s.nbDims; ++k) {
      if (static_cast<int32_t>(s.d[k]) != r.strides[k]) {
        return false;
      }
    }
  }
  return true;
}
#endif  // TACS_HAS_IOINFO

//---------- TacticsDB：tactics 记录的存储与查询 ------------

// 一条 tactic 记录（算法身份）。同一个结构 key 下可能对应多个不同
// (impl,tactic)：
//   - 富 key：record 模式下 TRT 对同 key 的多个层可能自由选到不同 tactic；
//   - OPAQUE 兜底 key：不同结构的融合 / 内部层塌到同一 shape key。
//selectAlgorithms 通过 "哪条记录的 (impl,tactic) 恰好出现在本层候选里" 来消歧，
// 在不依赖层名的前提下让每层仍选回属于它自己的 tactic。
struct TacticEntry {
  int64_t implementation{0};
  int64_t tactic{0};
  std::string layer_name;        // 诊断用：首次产生该 (impl,tactic) 的层名，不参与匹配
  int32_t hit_count{0};
  std::vector<IOFormat> ioinfo;  // 算法身份，供精确 tie-break
};

class TacticsDB {
 public:
  // 从 JSON 文件加载；不存在 / 解析失败 / 版本不符时静默保持空（触发 record 路径）
  void load(const std::string &path) {
    if (!file_exists(path)) {
      LOG(WARNING) << "[TacticsDB] file not found, start with empty DB:"
                   << path;
      return;
    }
    std::string content = get_file_content(path);
    std::string err;
    auto json = mjson::Json::parse(content, err);
    if (!err.empty() || !json.is_object()) {
      LOG(WARNING) << "[TacticsDB] parse error, start with empty DB:" << err;
      return;
    }
    // 版本校验：key / 格式 升级后旧 cache 不再兼容（保持空 DB → replay 全 miss →
    // 交 TRT 重新选并 record → save 时以新版本覆盖，自愈为新 cache）。
    std::string file_ver =
        json.has_key("version") ? json["version"].string_value() : "";
    if (file_ver != kTacticsCacheVersion) {
      LOG(WARNING) << "[TacticsDB] cache version mismatch (file='" << file_ver
                   << "', expected='" << kTacticsCacheVersion
                   << "'), ignoring stale cache and re-recording:" << path;
      return;
    }
    if (json.has_key("trt_version")) {
      int64_t ftrt =
          json["trt_version"].is_int64_t()
              ? json["trt_version"].int64_t_value()
              : static_cast<int64_t>(json["trt_version"].number_value());
      if (ftrt != TRT_VERSION_FOR_MINFER) {
        LOG(WARNING) << "[TacticsDB] cache built with TRT" << ftrt
                     << "but current is" << TRT_VERSION_FOR_MINFER
                     << "; tactics may not match (will fallback per layer).";
      }
    }
    if (!json.has_key("tactics") || !json["tactics"].is_object()) {
      LOG(WARNING) << "[TacticsDB] no 'tactics' object in file:" << path;
      return;
    }
    auto to_i64 = [](mjson::Json const &j) -> int64_t {
      return j.is_int64_t() ? j.int64_t_value()
                            : static_cast<int64_t>(j.number_value());
    };
    auto parse_entry = [&](mjson::Json const &val) -> TacticEntry {
      TacticEntry rec;
      rec.implementation = to_i64(val["impl"]);
      rec.tactic = to_i64(val["tactic"]);
      rec.layer_name = val["layer_name"].string_value();
      rec.hit_count = static_cast<int32_t>(val["hit_count"].number_value());
      if (val.has_key("ioinfo") && val["ioinfo"].is_array()) {
        for (auto const &io : val["ioinfo"].array_value()) {
          IOFormat f;
          f.format = static_cast<int32_t>(to_i64(io["fmt"]));
          f.dtype = static_cast<int32_t>(to_i64(io["dt"]));
          if (io.has_key("st") && io["st"].is_array()) {
            for (auto const &s : io["st"].array_value()) {
              f.strides.push_back(static_cast<int32_t>(to_i64(s)));
            }
          }
          f.vec_dim = to_i64(io["vd"]);
          f.comps = to_i64(io["cp"]);
          rec.ioinfo.push_back(std::move(f));
        }
      }
      return rec;
    };
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto const &kv : json["tactics"].object_value()) {
      // 每个 key 的 value 是候选记录数组。
      if (!kv.second.is_array()) {
        continue;
      }
      std::vector<TacticEntry> entries;
      for (auto const &val : kv.second.array_value()) {
        entries.push_back(parse_entry(val));
      }
      db_[kv.first] = std::move(entries);
    }
    LOG(WARNING) << "[TacticsDB] loaded " << db_.size() << " keys from "
                 << path;
  }

  void save(const std::string &path) {
    std::lock_guard<std::mutex> lock(mutex_);
    mjson::Json::object tactics_obj;
    for (auto const &kv : db_) {
      mjson::Json::array entries_arr;
      for (auto const &e : kv.second) {
        mjson::Json::object entry;
        entry["impl"] = mjson::Json(e.implementation);
        entry["tactic"] = mjson::Json(e.tactic);
        entry["layer_name"] = mjson::Json(e.layer_name);
        entry["hit_count"] = mjson::Json(static_cast<double>(e.hit_count));
        mjson::Json::array ioinfo_arr;
        for (auto const &io : e.ioinfo) {
          mjson::Json::object o;
          o["fmt"] = mjson::Json(static_cast<int64_t>(io.format));
          o["dt"] = mjson::Json(static_cast<int64_t>(io.dtype));
          mjson::Json::array st;
          for (int32_t s : io.strides) {
            st.emplace_back(mjson::Json(static_cast<int64_t>(s)));
          }
          o["st"] = mjson::Json(st);
          o["vd"] = mjson::Json(io.vec_dim);
          o["cp"] = mjson::Json(io.comps);
          ioinfo_arr.emplace_back(mjson::Json(o));
        }
        entry["ioinfo"] = mjson::Json(ioinfo_arr);
        entries_arr.emplace_back(mjson::Json(entry));
      }
      tactics_obj[kv.first] = mjson::Json(entries_arr);
    }
    mjson::Json::object root;
    root["version"] = mjson::Json(std::string(kTacticsCacheVersion));
    root["trt_version"] =
        mjson::Json(static_cast<int64_t>(TRT_VERSION_FOR_MINFER));
    root["note"] = mjson::Json(std::string(
        "tactics key (name-free): layerType[::prec=P]::in[dtype:shape]::out"
        "[dtype:shape][::config] when the algorithm context name directly "
        "resolves to a single ILayer in the current build; otherwise "
        "OPAQUE::in[dtype:shape]::out[dtype:shape]. name is only a per-build "
        "locator into the freshly-built layer map (duplicate names dropped), "
        "never stored/compared as identity. shape from context dims "
        "(algo-agnostic); dtype from ITensor::getType() when resolved else "
        "'AUTO'; config includes conv kernel/nom/stride/pad/dilation/groups "
        "and "
        "plugin type/version. Each key maps to an ARRAY of candidate records; "
        "selectAlgorithms disambiguates by which recorded (impl,tactic) is "
        "present in the layer's choices, then ioinfo tie-break. ioinfo "
        "(tensorFormat/dataType/strides/vectorizedDim/componentsPerElement) is "
        "algorithm-derived, stored only for tie-break, NOT part of the key; "
        "impl/tactic stored as exact int64."));
    root["tactics"] = mjson::Json(tactics_obj);
    std::string json_str;
    mjson::Json(root).dump(json_str);
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
      LOG(WARNING) << "[TacticsDB] failed to open for write: " << path;
      return;
    }
    ofs << json_str;
    LOG(WARNING) << "[TacticsDB] saved " << db_.size() << " keys to " << path;
  }

  // 查找 key；命中返回候选记录列表指针，未命中返回 nullptr。
  const std::vector<TacticEntry> *query(const std::string &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = db_.find(key);
    if (it == db_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  // 记录：同一 key 下若已存在相同 (impl,tactic) 则只累加 hit_count，否则追加为
  // 新的候选记录（多结构塌到同一兜底 key 时，各自的 tactic 都会被保留）。
  //only_new_key=true（replay 模式）：仅当 key 完全不存在时才建首条记录（miss
  // 自愈），已有 key 一律不追加 —— 避免回退挑到的 tactic 污染 cache、避免
  //replay 之间 cache 漂移。only_new_key=false（record
  // 模式）：正常累积所有变体。
  void record(const std::string &key, const std::string &layer_name,
              int64_t impl, int64_t tactic, std::vector<IOFormat> ioinfo,
              bool only_new_key = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (only_new_key && db_.find(key) != db_.end()) {
      return;
    }
    auto &entries = db_[key];
    for (auto &e : entries) {
      if (e.implementation == impl && e.tactic == tactic) {
        e.hit_count++;
        return;
      }
    }
    TacticEntry rec;
    rec.implementation = impl;
    rec.tactic = tactic;
    rec.layer_name = layer_name;
    rec.hit_count = 0;
    rec.ioinfo = std::move(ioinfo);
    entries.push_back(std::move(rec));
    LOG(WARNING) << "[TacticsDB] new record: " << key << " (impl=" << impl
                 << ",tactic=" << tactic << ", variants=" << entries.size()
                 << ")";
  }

 private:
  std::unordered_map<std::string, std::vector<TacticEntry>> db_;
  std::mutex mutex_;
};

//---------- TacticsSelector：IAlgorithmSelector 实现 ----------

class TacticsSelector : public nvinfer1::IAlgorithmSelector {
 public:
  TacticsSelector(TacticsDB &db, bool record_mode, TacsLayerMap layer_map)
      : db_(db), record_mode_(record_mode), layer_map_(std::move(layer_map)) {}

  //record mode：全部 return 0，TRT 自由选（记录其计时最优选择）。
  //replay mode：查 DB —— 命中则在候选里找 " (impl,tactic)
  // 与某条记录一致 " 的候选并强制选中（多条命中用 ioinfo
  //tie-break，再退确定性最小）；tactic 不在候选里 → 确定性回退；miss（key
  // 未记录）→ return 0 交 TRT 自由选并在 report 阶段补录。
  int32_t selectAlgorithms(nvinfer1::IAlgorithmContext const &context,
                           nvinfer1::IAlgorithm const *const *choices,
                           int32_t nb_choices,
                           int32_t *selected) noexcept override {
    if (record_mode_ || nb_choices == 0) {
      return 0;
    }
    std::string key = tacs_extract_key(context, *choices[0], layer_map_);
    if (key.empty()) {
      return 0;
    }
    const std::vector<TacticEntry> *entries = db_.query(key);
    if (entries == nullptr || entries->empty()) {
      //miss：未记录的层（正常 cache + 同模型应为 0）。交 TRT 选并 report
      // 补录， 下次 replay 即可 pin。稳态无 miss，可复现。
      LOG(WARNING) << "[TacticsSelector] miss (fallback to TRT):" << key;
      return 0;
    }

    int32_t nb_io = context.getNbInputs() + context.getNbOutputs();
    (void)nb_io;
    // 收集：候选中 (impl,tactic) 命中任一记录的下标（下标升序）。
    std::vector<int32_t> matched;
    for (int32_t i = 0; i < nb_choices; ++i) {
      auto const &v = choices[i]->getAlgorithmVariant();
      for (auto const &e : *entries) {
        if (v.getImplementation() == e.implementation &&
            v.getTactic() == e.tactic) {
          matched.push_back(i);
          break;
        }
      }
    }
    if (!matched.empty()) {
      int32_t chosen = -1;

#if TACS_HAS_IOINFO
      // 优先：候选的完整 IO format 与其对应记录的 ioinfo
      // 一致（精确复现同一变体）。
      for (int32_t idx : matched) {
        auto const &v = choices[idx]->getAlgorithmVariant();
        bool io_match = false;
        for (auto const &e : *entries) {
          if (v.getImplementation() == e.implementation &&
              v.getTactic() == e.tactic && !e.ioinfo.empty() &&
              tacs_ioinfo_matches(*choices[idx], e.ioinfo, nb_io)) {
            io_match = true;
            break;
          }
        }
        if (!io_match) {
          continue;
        }
        if (chosen < 0) {
          chosen = idx;
          continue;
        }
        auto const &vc = choices[chosen]->getAlgorithmVariant();
        if (v.getImplementation() < vc.getImplementation() ||
            (v.getImplementation() == vc.getImplementation() &&
             v.getTactic() < vc.getTactic())) {
          chosen = idx;
        }
      }
#endif  // TACS_HAS_IOINFO
      if (chosen < 0) {
        // 确定性：matched 里取 (impl,tactic) 最小（与候选数组顺序无关）。
        chosen = matched[0];
        for (int32_t idx : matched) {
          auto const &vi = choices[idx]->getAlgorithmVariant();
          auto const &vc = choices[chosen]->getAlgorithmVariant();
          if (vi.getImplementation() < vc.getImplementation() ||
              (vi.getImplementation() == vc.getImplementation() &&
               vi.getTactic() < vc.getTactic())) {
            chosen = idx;
          }
        }
      }
      selected[0] = chosen;
      LOG(WARNING) << "[TacticsSelector] hit:" << key;
      return 1;
    }

    // 诊断：打出记录的候选 (impl,tactic) 和本次全部候选。
    {
      std::ostringstream rec_oss;
      for (size_t k = 0; k < entries->size(); ++k) {
        if (k > 0) {
          rec_oss << ",";
        }
        rec_oss << "(impl=" << (*entries)[k].implementation
                << ",tactic=" << (*entries)[k].tactic << ")";
      }
      std::ostringstream choices_oss;
      for (int32_t i = 0; i < nb_choices; ++i) {
        auto const &v = choices[i]->getAlgorithmVariant();
        if (i > 0) {
          choices_oss << ",";
        }
        choices_oss << "(impl=" << v.getImplementation()
                    << ",tactic=" << v.getTactic() << ")";
      }
      LOG(WARNING) << "[TacticsSelector] tactic not in choices (fallback):"
                   << key << "recorded=[" << rec_oss.str() << "] choices=["
                   << choices_oss.str() << "]";
    }
    // 记录的 tactic 都不在本层候选里。按 (impl,tactic) 最小者确定性选择（与
    //choices 顺序无关），保证每次 build 该层都选同一候选。不 record（key
    // 已存在），不污染 cache。
    int32_t best = 0;
    for (int32_t i = 1; i < nb_choices; ++i) {
      auto const &vi = choices[i]->getAlgorithmVariant();
      auto const &vb = choices[best]->getAlgorithmVariant();
      if (vi.getImplementation() < vb.getImplementation() ||
          (vi.getImplementation() == vb.getImplementation() &&
           vi.getTactic() < vb.getTactic())) {
        best = i;
      }
    }
    selected[0] = best;
    LOG(WARNING) << "[TacticsSelector] deterministic fallback:" << key
                 << "-> (impl="
                 << choices[best]->getAlgorithmVariant().getImplementation()
                 << ",tactic="
                 << choices[best]->getAlgorithmVariant().getTactic() << ")";
    return 1;
  }

  // 两种 mode 均调用 db_.record()：record mode 收录所有层，replay mode 追加新层
  // 或同 key 的新 (impl,tactic) 变体。
  void reportAlgorithms(nvinfer1::IAlgorithmContext const *const *contexts,
                        nvinfer1::IAlgorithm const *const *choices,
                        int32_t nb) noexcept override {
    for (int32_t i = 0; i < nb; ++i) {
      std::string key = tacs_extract_key(*contexts[i], *choices[i], layer_map_);
      if (key.empty()) {
        continue;
      }
      std::string layer_name = contexts[i]->getName();
      auto const &v = choices[i]->getAlgorithmVariant();
      std::vector<IOFormat> ioinfo;
#if TACS_HAS_IOINFO
      int32_t nb_io = contexts[i]->getNbInputs() + contexts[i]->getNbOutputs();
      ioinfo = tacs_collect_ioinfo(*choices[i], nb_io);
#endif  // TACS_HAS_IOINFO
      //replay 模式（!record_mode_）：只对全新 key 自愈，不给已有 key
      // 追加变体， 保证 replay 不改动 cache。record 模式：累积全部变体。
      db_.record(key, layer_name, v.getImplementation(), v.getTactic(),
                 std::move(ioinfo), /*only_new_key=*/!record_mode_);

      LOG(WARNING) << "[TacticsDiag] name=" << layer_name << " key=" << key
                   << " impl=" << v.getImplementation()
                   << " tactic=" << v.getTactic();
    }
  }

 private:
  TacticsDB &db_;
  bool record_mode_;
  TacsLayerMap layer_map_;
};

}  // namespace minfer

#endif  // TRT_VERSION_FOR_MINFER >= 8000 && TRT_VERSION_FOR_MINFER < 10000
