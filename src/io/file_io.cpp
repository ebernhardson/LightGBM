#include <LightGBM/utils/log.h>
#include <LightGBM/utils/file_io.h>

#include <algorithm>
#include <sstream>
#include <unordered_map>
#ifdef USE_HDFS
#include <hdfs.h>
#endif

namespace LightGBM{

struct LocalFile : VirtualFileReader, VirtualFileWriter {
  LocalFile(const Uri& uri, const std::string& mode) : uri_(uri), mode_(mode) {}
  virtual ~LocalFile() {
    if (file_ != NULL) {
      fclose(file_);
    }
  }

  bool Init() {
    if (file_ == NULL) {
#if _MSC_VER
      fopen_s(&file_, uri_.name.c_str(), mode_.c_str());
#else
      file_ = fopen(uri_.name.c_str(), mode_.c_str());
#endif
    }
    return file_ != NULL;
  }

  bool Exists() const {
    LocalFile file(uri_.name, "rb");
    return file.Init();
  }

  size_t Read(void* buffer, size_t bytes) const {
    return fread(buffer, 1, bytes, file_);
  }

  size_t Write(const void* buffer, size_t bytes) const {
    return fwrite(buffer, bytes, 1, file_) == 1 ? bytes : 0;
  }

private:
  FILE* file_ = NULL;
  const Uri uri_;
  const std::string mode_;
};

struct MultiFileReader : VirtualFileReader {
  MultiFileReader(const Uri& uri) : uri_(uri) {
    size_t start = 0;
    while (true) {
      size_t end = uri.uri.find(',', start);
      std::string filename = uri.uri.substr(start, end - start);
      if (!filename.empty()) {
        filename += uri.suffix;
        filenames_.push_back(filename);
      }
      if (end == std::string::npos) {
          break;
      }
      start = end + 1;
    }
  }

  bool Init() {
    if (position_ >= filenames_.size()) {
        return false;
    }
    if (reader_ == nullptr) {
        reader_ = VirtualFileReader::Make(filenames_.at(position_));
    }
    if (reader_->Init()) {
        return true;
    } else {
        reader_ = nullptr;
        return false;
    }
  }

  bool Exists() const {
    return VirtualFileWriter::Exists(filenames_.at(0));
  }

  size_t Read(void* data, size_t bytes) const {
    if (position_ >= filenames_.size() || reader_ == nullptr) {
      return 0;
    }
    char* buffer = (char*)data;
    size_t nleft = bytes;
    size_t nread = 0;
    while (true) {
      size_t nread_i = reader_->Read(buffer, nleft);
      buffer += nread_i;
      nleft -= nread_i;
      nread += nread_i;
      if (nleft == 0) {
        return nread;
      }
      if (nread_i == 0) {
        position_ += 1;
        if (position_ >= filenames_.size()) {
          return nread;
        }
        reader_ = VirtualFileReader::Make(filenames_.at(position_));
        if (!reader_->Init()) {
           Log::Fatal("...");
        }
      }
    }
  }

private:
  Uri uri_;
  std::vector<std::string> filenames_;
  mutable size_t position_ = 0;
  mutable std::unique_ptr<VirtualFileReader> reader_ = nullptr;
};

const std::string kHdfsProto = "hdfs://";

#ifdef USE_HDFS
struct HdfsFile : VirtualFileReader, VirtualFileWriter {
  HdfsFile(const Uri& uri, int flags) : uri_(uri), flags_(flags) {}
  ~HdfsFile() {
    if (file_ != NULL) {
      hdfsCloseFile(fs_, file_);
    }
  }

  bool Init() {
    if (file_ == NULL) {
      if (fs_ == NULL) {
        fs_ = getHdfsFS(uri_);
      }
      if (fs_ != NULL && (flags_ == O_WRONLY || 0 == hdfsExists(fs_, uri_.name.c_str()))) {
        file_ = hdfsOpenFile(fs_, uri_.name.c_str(), flags_, 0, 0, 0);
      }
    }
    return file_ != NULL;
  }

  bool Exists() const {
    if (fs_ == NULL) {
      fs_ = getHdfsFS(uri_);
    }
    return fs_ != NULL && 0 == hdfsExists(fs_, uri_.name.c_str());
  }

  size_t Read(void* data, size_t bytes) const {
    return FileOperation<void*>(data, bytes, &hdfsRead);
  }

  size_t Write(const void* data, size_t bytes) const {
    return FileOperation<const void*>(data, bytes, &hdfsWrite);
  }

private:
  template <typename BufferType>
  using fileOp = tSize(*)(hdfsFS, hdfsFile, BufferType, tSize);

  template <typename BufferType>
  inline size_t FileOperation(BufferType data, size_t bytes, fileOp<BufferType> op) const {
    char* buffer = (char *)data;
    size_t remain = bytes;
    while (remain != 0) {
      size_t nmax = static_cast<size_t>(std::numeric_limits<tSize>::max());
      tSize ret = op(fs_, file_, buffer, std::min(nmax, remain));
      if (ret > 0) {
        size_t n = static_cast<size_t>(ret);
        remain -= n;
        buffer += n;
      } else if (ret == 0) {
        break;
      } else if (errno != EINTR) {
        Log::Fatal("Failed hdfs file operation [%s]", strerror(errno));
      }
    }
    return bytes - remain;
  }

  static hdfsFS getHdfsFS(const Uri& uri) {
    size_t end = uri.uri.find("/", kHdfsProto.length());
    if (uri.uri.find(kHdfsProto) != 0 || end == std::string::npos) {
      Log::Warning("Bad hdfs uri, no namenode found [%s]", uri.name.c_str());
      return NULL;
    }
    std::string hostport = uri.uri.substr(kHdfsProto.length(), end - kHdfsProto.length());
    if (fs_cache_.count(hostport) == 0) {
      fs_cache_[hostport] = makeHdfsFs(hostport);
    }
    return fs_cache_[hostport];
  }

  static hdfsFS makeHdfsFs(const std::string& hostport) {
    std::istringstream iss(hostport);
    std::string host;
    tPort port = 0;
    std::getline(iss, host, ':');
    iss >> port;
    hdfsFS fs = iss.eof() ? hdfsConnect(host.c_str(), port) : NULL;
    if (fs == NULL) {
      Log::Warning("Could not connect to hdfs namenode [%s]", hostport.c_str());
    }
    return fs;
  }

  mutable hdfsFS fs_ = NULL;
  hdfsFile file_ = NULL;
  const Uri uri_;
  const int flags_;
  static std::unordered_map<std::string, hdfsFS> fs_cache_;
};

std::unordered_map<std::string, hdfsFS> HdfsFile::fs_cache_ = std::unordered_map<std::string, hdfsFS>();

#define WITH_HDFS(x) x
#else
#define WITH_HDFS(x) Log::Fatal("HDFS Support not enabled.")
#endif // USE_HDFS

std::unique_ptr<VirtualFileReader> VirtualFileReader::Make(const Uri& uri) {
  if (uri.uri.find(',') != std::string::npos) {
    return std::unique_ptr<VirtualFileReader>(new MultiFileReader(uri));
  } else if (0 == uri.uri.find(kHdfsProto)) {
    WITH_HDFS(return std::unique_ptr<VirtualFileReader>(new HdfsFile(uri, O_RDONLY)));
  } else {
    return std::unique_ptr<VirtualFileReader>(new LocalFile(uri, "rb"));
  }
}

std::unique_ptr<VirtualFileWriter> VirtualFileWriter::Make(const Uri& uri) {
  if (uri.uri.find(',') != std::string::npos) {
    Log::Fatal("Multi-file not supported for writes");
  } else if (0 == uri.uri.find(kHdfsProto)) {
    WITH_HDFS(return std::unique_ptr<VirtualFileWriter>(new HdfsFile(uri, O_WRONLY)));
  } else {
    return std::unique_ptr<VirtualFileWriter>(new LocalFile(uri, "wb"));
  }
}

bool VirtualFileWriter::Exists(const Uri& uri) {
  if (0 == uri.uri.find(kHdfsProto)) {
    WITH_HDFS(HdfsFile file(uri, O_RDONLY); return file.Exists());
  } else {
      LocalFile file(uri, "rb");
      return file.Exists();
  }
}

}  // namespace LightGBM
