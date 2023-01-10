#pragma once

#include "hetu/common/macros.h"
#include <tuple>

namespace hetu {

enum class DeviceType : int8_t {
  CPU = 0,
  CUDA,
  NUM_DEVICE_TYPES,
  UNDETERMINED
};

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kUndeterminedDevice = DeviceType::UNDETERMINED;
constexpr int16_t NUM_DEVICE_TYPES =
  static_cast<int16_t>(DeviceType::NUM_DEVICE_TYPES);

std::string DeviceType2Str(const DeviceType&);
std::ostream& operator<<(std::ostream&, const DeviceType&);

using DeviceIndex = uint8_t;

#define HT_MAX_DEVICE_INDEX (16)
#define HT_MAX_HOSTNAME_LENGTH (256)
#define HT_MAX_DEVICE_MULTIPLEX (16)

class Device {
 public:
  static constexpr char BACK_SLASH = '/';
  static constexpr char COLON = ':';

  Device(DeviceType type = kUndeterminedDevice, DeviceIndex index = 0U,
         const std::string& hostname = "", uint8_t multiplex = 0U) {
    _init(type, index, hostname, multiplex);
  }

  Device(const std::string& device, uint8_t multiplex = 0U);

  Device(const Device&) = default;
  Device(Device&&) = default;
  Device& operator=(const Device& device) = default;
  Device& operator=(Device&& device) = default;

  ~Device() = default;

  inline bool operator==(const Device& device) const {
    return type() == device.type() && index() == device.index() &&
      hostname() == device.hostname() && multiplex() == device.multiplex();
  }

  inline bool operator!=(const Device& device) const {
    return !operator==(device);
  }

  inline bool operator<(const Device& device) const {
    if (hostname() != device.hostname())
      return hostname() < device.hostname();
    if (type() != device.type())
      return type() < device.type();
    if (index() != device.index())
      return index() < device.index();
    if (multiplex() != device.multiplex())
      return multiplex() < device.multiplex();
    return false;
  }

  inline DeviceType type() const noexcept {
    return _type;
  }

  inline DeviceIndex index() const noexcept {
    return _index;
  }

  inline bool is_cpu() const noexcept {
    return _type == kCPU;
  }

  inline bool is_cuda() const noexcept {
    return _type == kCUDA;
  }

  inline bool is_undetermined() const noexcept {
    return _type == kUndeterminedDevice;
  }

  inline bool local() const noexcept {
    return _hostname.empty();
  }

  inline const std::string& hostname() const noexcept {
    return _hostname;
  }

  inline uint8_t multiplex() const noexcept {
    return _multiplex;
  }

  std::string compat_string() const;

  static inline std::string GetLocalHostname() {
    char* env = std::getenv("HETU_LOCAL_HOSTNAME");
    if (env == nullptr) {
      return "";
    } else {
      std::string ret(env);
      HT_ASSERT(ret.length() <= HT_MAX_HOSTNAME_LENGTH)
        << "Hostname \"" << ret << "\" exceeds max length "
        << HT_MAX_HOSTNAME_LENGTH;
      ;
      return ret;
    }
  }

 private:
  void _init(DeviceType type, DeviceIndex index, const std::string& hostname,
             uint8_t multiplex) {
    _type = type;
    _index = _type == kCUDA ? index : 0U;
    HT_ASSERT(_index < HT_MAX_DEVICE_INDEX)
      << "Device index " << _index << " exceeds maximum allowed value "
      << HT_MAX_DEVICE_INDEX;
    if (!hostname.empty() && hostname != "localhost" &&
        hostname != GetLocalHostname()) {
      HT_ASSERT(hostname.length() <= HT_MAX_HOSTNAME_LENGTH)
        << "Hostname \"" << hostname << "\" exceeds max length "
        << HT_MAX_HOSTNAME_LENGTH;
      _hostname = hostname;
    }
    HT_ASSERT(multiplex < HT_MAX_DEVICE_MULTIPLEX)
      << "Multiplex " << multiplex << " exceeds maximum allowed value "
      << HT_MAX_HOSTNAME_LENGTH;
    _multiplex = multiplex;
  }

  DeviceType _type;
  DeviceIndex _index;
  std::string _hostname;
  uint8_t _multiplex;
};

std::ostream& operator<<(std::ostream&, const Device&);

class DeviceGroup {
 public:
  DeviceGroup(const std::vector<Device>& devices) : _devices(devices) {
    std::sort(_devices.begin(), _devices.end());
    _devices.erase(std::unique(_devices.begin(), _devices.end()),
                   _devices.end());
  }

  DeviceGroup(const std::vector<std::string>& devices) {
    _devices.reserve(devices.size());
    for (const auto& device : devices)
      _devices.emplace_back(device);
    std::sort(_devices.begin(), _devices.end());
    _devices.erase(std::unique(_devices.begin(), _devices.end()),
                   _devices.end());
  }

  DeviceGroup() : DeviceGroup(std::vector<Device>()) {}

  DeviceGroup(const DeviceGroup&) = default;
  DeviceGroup& operator=(const DeviceGroup&) = default;
  DeviceGroup(DeviceGroup&&) = default;
  DeviceGroup& operator=(DeviceGroup&&) = default;

  inline bool operator==(const DeviceGroup& device_group) const {
    return _devices == device_group._devices;
  }

  inline bool operator!=(const DeviceGroup& device_group) const {
    return !operator==(device_group);
  }

  inline bool operator<(const DeviceGroup& device_group) const {
    return _devices < device_group._devices;
  }

  inline size_t get_index(const Device& device) const {
    auto it = std::find(_devices.begin(), _devices.end(), device);
    HT_ASSERT_NE(it, _devices.end()) << "Device not found: " << device;
    return it - _devices.begin();
  }

  inline const std::vector<Device>& devices() const {
    return _devices;
  }

  inline size_t num_devices() const {
    return _devices.size();
  }

  inline bool empty() const {
    return _devices.empty();
  }

  inline bool contains(const Device& device) const {
    auto it = std::find(_devices.begin(), _devices.end(), device);
    return it != _devices.end();
  }

  inline const Device& get(size_t i) const {
    return _devices[i];
  }

  inline void set(size_t i, Device device) {
    _devices[i] = device;
  }

 private:
  std::vector<Device> _devices;
};

std::ostream& operator<<(std::ostream&, const DeviceGroup&);

} // namespace hetu

namespace std {

template <>
struct hash<hetu::Device> {
  std::size_t operator()(const hetu::Device& device) const noexcept {
    auto hash = std::hash<int>()((static_cast<int>(device.type()) << 16) |
                                 (static_cast<int>(device.index()) << 8) |
                                 static_cast<int>(device.multiplex()));
    if (!device.local()) {
      // Following boost::hash_combine
      hash ^= (std::hash<std::string>()(device.hostname()) + 0x9e3779b9 +
               (hash << 6) + (hash >> 2));
    }
    return hash;
  }
};

template <>
struct hash<hetu::DeviceGroup> {
  std::size_t operator()(const hetu::DeviceGroup& group) const noexcept {
    // devices in group are sorted, so we can hash them without re-ordering
    const auto& devices = group.devices();
    auto hash_op = std::hash<hetu::Device>();
    auto seed = devices.size();
    for (const auto& device : devices)
      seed ^= hash_op(device) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

inline std::string to_string(const hetu::Device& device) {
  std::ostringstream os;
  os << device;
  return os.str();
}

inline std::string to_string(const hetu::DeviceGroup& group) {
  std::ostringstream os;
  os << group;
  return os.str();
}

} // namespace std