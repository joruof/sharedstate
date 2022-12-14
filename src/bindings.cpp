#include <cstring>
#include <ios>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <unordered_map>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/**
 * Storage format:
 *
 * list of object types:
 *
 * type_count: uint32
 *
 *   total_size: uint32
 *   type_name_len: uint32, type_name: str
 *   attrib_count: uint32
 *     attrib_name_len: uint32, attrib_name: str
 *     attrib_type: uint8
 *     attrib_name_len: uint32, attrib_name: str
 *     attrib_type: uint8
 *     attrib_name_len: uint32, attrib_name: str
 *     attrib_type: uint8
 *     ...
 *  ...
 *
 * object_tree:
 *
 * main_type_name_len: uint32, main_type_name: str
 * object data
 * ...
 *
 */

/**
 * Thin wrapper around the file lock (or rather range lock) concept.
 */
struct RangeLock {
    
    const int descriptor;
    const int offset;
    const size_t len;

    RangeLock (int descriptor, int offset, size_t len) :
        descriptor{descriptor}, offset{offset}, len{len}  {
    }

    flock buildFlock(short type) {

        flock lock;

        lock.l_whence = SEEK_SET;
        lock.l_start = offset;
        lock.l_len = len;
        lock.l_pid = 0;
        lock.l_type = type;

        return lock;
    }

    int lock (short type, bool blocking = true) { 

        flock lock = buildFlock(type);

        if (-1 == fcntl(descriptor, blocking ? F_OFD_SETLKW : F_OFD_SETLK , &lock)) {
            return errno;
        }

        return 0;
    }

    int unlock () {

        flock lock = buildFlock(F_UNLCK);

        if (-1 == fcntl(descriptor, F_OFD_SETLK, &lock)) {
            return errno;
        }

        return 0;
    }

    bool check (short type) {

        flock lock = buildFlock(type);

        if (-1 == fcntl(descriptor, F_OFD_GETLK, &lock)) {
            return false;
        }

        return lock.l_type == F_UNLCK;
    }
};

template <typename T>
using array_like = py::array_t<T, py::array::c_style | py::array::forcecast>;

enum Kind {

    KIND_NONE,
    KIND_INT,
    KIND_FLOAT,
    KIND_BOOL,
    KIND_STRING,
    KIND_ARRAY,
    KIND_STRUCT
};

struct Field {
    
    Kind kind = KIND_NONE;

    std::string name;

    // length of an instance of this field in byte
    size_t length = 0;

    // relative offset to the object start address in byte
    size_t offset = 0;

    // only used for structs, otherwise empty
    // contains the name of the struct type in this field
    std::string ref_type_name;

    Field () {
    }

    Field (Kind kind,
           std::string name,
           size_t length,
           size_t offset) 
        : Field(kind, name, length, offset, "") {
    }

    Field (Kind kind,
           std::string name,
           size_t length,
           size_t offset,
           std::string ref_type_name) 
        : kind{kind},
          name{name},
          length{length},
          offset{offset}, 
          ref_type_name{ref_type_name} {
    }
};

struct ObjectType {

    std::vector<Field> fields;

    // maps from field names to
    std::unordered_map<std::string, size_t> field_map;

    // the length of an instance of this type in bytes
    size_t length = 0;

    Field& operator[] (std::string field_name) {

        return fields[field_map[field_name]];
    }
};

using TypeDict = std::unordered_map<std::string, ObjectType>;

/**
 * Returns the fully qualified typename of a given object.
 */
std::string get_fq_type_name (py::object& obj) {

    // get qualified name of the object type

    py::object obj_class = obj.attr("__class__");
    py::object obj_module = obj.attr("__module__");

    std::string obj_type_qualname = py::str(obj_class.attr("__qualname__"));
    std::string obj_module_name = py::str(obj_module);

    if (obj_module_name != "builtins") {
        obj_type_qualname = obj_module_name + "." + obj_type_qualname;
    }

    return obj_type_qualname;
}

/**
 * This extracts the type structure from the provided python object tree.
 */
size_t build_type_dict (TypeDict& type_dict, py::object& obj) {

    std::string obj_type_qualname = get_fq_type_name(obj);

    if (type_dict.count(obj_type_qualname) > 0) {
        // we already have that type in the dict
        return type_dict[obj_type_qualname].length;
    }
    
    ObjectType& obj_type = type_dict[obj_type_qualname];

    py::dict obj_dict = obj.attr("__dict__");

    size_t offset = 0;

    for (auto kv : obj_dict) {

        std::string field_name = py::str(kv.first);
        if (field_name[0] == '_') {
            continue;
        }

        py::handle f = kv.second;

        std::string field_type_name = py::str(f.get_type().attr("__name__"));

        if (field_type_name == "int") {
            obj_type.field_map[field_name] = obj_type.fields.size();
            obj_type.fields.emplace_back(KIND_INT, field_name, 8, offset);
            offset += 8;
        } else if (field_type_name == "float") {
            obj_type.field_map[field_name] = obj_type.fields.size();
            obj_type.fields.emplace_back(KIND_FLOAT, field_name, 8, offset);
            offset += 8;
        } else if (field_type_name == "str") {
            obj_type.field_map[field_name] = obj_type.fields.size();
            obj_type.fields.emplace_back(KIND_STRING, field_name, 4 + 256, offset);
            offset += 4 + 256;
        } else if (field_type_name == "bool") {
            obj_type.field_map[field_name] = obj_type.fields.size();
            obj_type.fields.emplace_back(KIND_BOOL, field_name, 8, offset);
            offset += 1;
        } else if (field_type_name == "ndarray") {
            array_like<double> arr = array_like<double>::ensure(f);
            // 8 bytes for ndim, 8 bytes per dim, array length
            size_t len = 8 + arr.ndim() * 8 + arr.nbytes();
            obj_type.field_map[field_name] = obj_type.fields.size();
            obj_type.fields.emplace_back(KIND_ARRAY, field_name, len, offset);
            offset += len;
        } else if (py::hasattr(f, "__dict__")) {
            py::object fo = py::reinterpret_borrow<py::object>(f);
            obj_type.field_map[field_name] = obj_type.fields.size();
            obj_type.fields.emplace_back(
                    KIND_STRUCT,
                    field_name,
                    8,
                    offset,
                    get_fq_type_name(fo));
            offset += build_type_dict(type_dict, fo);
        } else {
            std::string msg = "field type \"" 
                + field_type_name + "\" of class \"" 
                + obj_type_qualname + "\" not supported";
            throw py::value_error(msg);
        }
    }

    obj_type.length = offset;
    
    return offset;
}

/**
 * Writing helper functions
 */

template<typename T>
void write_val (uint8_t*& mem, T value) {

    *((T*)mem) = value;
    mem += sizeof(T);
}

void write_val (uint8_t*& mem, const std::string& value) {

    // string length
    write_val<uint32_t>(mem, value.size());

    std::memcpy(mem, value.c_str(), value.size());
    mem += sizeof(uint8_t) * value.size();
}

void write_val (uint8_t*& mem, Field& value) {

    write_val<Kind>(mem, value.kind);
    write_val(mem, value.name);
    write_val(mem, value.length);
    write_val(mem, value.offset);

    if (value.kind == KIND_STRUCT) {
        write_val(mem, value.ref_type_name);
    }
}

/**
 * This writes the given type dict as the type header at the
 * beginning of a shared memory region. Returns a pointer to
 * after the type header.
 */
uint8_t* write_type_dict (TypeDict& type_dict, uint8_t* mem) {

    write_val<size_t>(mem, type_dict.size());

    for (auto kv: type_dict) {

        const std::string& type_name = kv.first;
        ObjectType& ot = kv.second;

        write_val(mem, type_name);
        write_val(mem, ot.length);

        write_val(mem, ot.fields.size());

        for (Field& f: ot.fields) {
            write_val(mem, f);
        }
    }

    return mem;
}

/**
 * Calculates the size of the serialized type dict.
 */
size_t type_dict_size (TypeDict& type_dict) {

    size_t tds = 0;

    tds += sizeof(type_dict.size());

    for (auto kv: type_dict) {

        const std::string& type_name = kv.first;
        ObjectType& ot = kv.second;

        tds += sizeof(uint32_t);
        tds += type_name.length();

        tds += sizeof(ot.fields.size());

        for (Field& f: ot.fields) {

            tds += sizeof(f.kind);

            tds += sizeof(uint32_t);
            tds += f.name.length();

            tds += sizeof(f.length);
            tds += sizeof(f.offset);

            if (f.kind == KIND_STRUCT) {
                tds += sizeof(uint32_t);
                tds += f.ref_type_name.length();
            }
        }
    }

    return tds;
}

/**
 * Reading helper functions
 */

template<typename T>
void read_val (uint8_t*& mem, T& value) {

    value = *((T*)mem);
    mem += sizeof(T);
}

void read_val (uint8_t*& mem, std::string& value) {

    // string length
    uint32_t len = 0;
    read_val(mem, len);

    value = std::string((char*)mem, len);
    mem += sizeof(uint8_t) * len;
}

void read_val (uint8_t*& mem, Field& value) {

    read_val(mem, value.kind);
    read_val(mem, value.name);
    read_val(mem, value.length);
    read_val(mem, value.offset);

    if (value.kind == KIND_STRUCT) {
        read_val(mem, value.ref_type_name);
    } 
}

/**
 * This reads and constructs a type dict from the type header.
 * Returns a pointer to right after the type dict.
 */
uint8_t* read_type_dict (TypeDict& type_dict, uint8_t* mem) {

    size_t type_count = 0;
    read_val(mem, type_count);

    for (size_t i = 0; i < type_count; ++i) {

        std::string type_name;
        read_val(mem, type_name);

        ObjectType& ot = type_dict[type_name];
        read_val(mem, ot.length);

        size_t field_count = 0;
        read_val(mem, field_count);

        for (uint32_t j = 0; j < field_count; ++j) {
            size_t pos = ot.fields.size();
            ot.fields.emplace_back();
            Field& f = ot.fields.back();
            read_val(mem, f);
            ot.field_map[f.name] = pos;
        }
    }

    return mem;
}

void write_object (
        uint8_t* mem,
        TypeDict& type_dict,
        py::object& obj,
        std::string& obj_fqtn);

void write_field (
        uint8_t* mem,
        TypeDict& type_dict,
        Field& field,
        py::object& obj_val) {

    uint8_t* ptr = mem + field.offset;

    switch (field.kind) {
        case KIND_INT: {
            int64_t val = py::int_(obj_val);
            write_val(ptr, val);
            break;
        }
        case KIND_FLOAT: {
            double val = py::float_(obj_val);
            write_val(ptr, val);
            break;
        }
        case KIND_STRING: {
            std::string val = py::str(obj_val);
            if (val.length() > 256) {
                throw py::value_error("maximum string length of 256 chars exceeded");
            }
            write_val(ptr, val);
            break;
        }
        case KIND_BOOL: {
            uint8_t val = (uint8_t)py::bool_(obj_val);
            write_val(ptr, val);
            break;
        }
        case KIND_ARRAY: {
            array_like<double> val = array_like<double>::ensure(obj_val);
            size_t length = 8 + 8 * val.ndim() + val.nbytes();
            if (length > field.length) {
                throw py::value_error("setting array of " 
                        + std::to_string(length)
                        + " bytes exceeds allowed size of " 
                        + std::to_string(field.length)
                        + " bytes");
            }
            // writes shape 
            write_val<ssize_t>(ptr, val.ndim());
            for (ssize_t k = 0; k < val.ndim(); ++k) {
                write_val<ssize_t>(ptr, val.shape()[k]);
            }
            // writes array contents
            std::memcpy(ptr, val.data(), val.nbytes());
            break;
        }
        case KIND_STRUCT: {
            write_object(ptr, type_dict, obj_val, field.ref_type_name);
            break;
        }
        default:
            return;
    };
}

void write_object (
        uint8_t* mem,
        TypeDict& type_dict,
        py::object& obj,
        std::string& obj_fqtn) {

    ObjectType& ot = type_dict[obj_fqtn];

    py::dict obj_dict = obj.attr("__dict__");

    for (Field& field : ot.fields) {

        if (!py::hasattr(obj, field.name.c_str())) {
            // this is debateable, i like it
            continue;
        }

        py::object obj_val = py::getattr(obj, field.name.c_str());

        write_field(mem, type_dict, field, obj_val);
    }
}

struct StoreObject {

    int shm_descriptor;

    uint8_t* memory_start;
    uint8_t* object_memory;

    std::string obj_type_qualname;
    ObjectType& obj_type;
    TypeDict& type_dict;

    RangeLock mem_lock;

    std::shared_ptr<std::vector<uint8_t>> own_data = nullptr;

    StoreObject (
            int shm_descriptor,
            uint8_t* memory_start,
            uint8_t* object_memory,
            std::string obj_type_qualname,
            ObjectType& obj_type,
            TypeDict& type_dict) 
        : shm_descriptor{shm_descriptor},
          memory_start{memory_start},
          object_memory{object_memory},
          obj_type_qualname{obj_type_qualname},
          obj_type{obj_type},
          type_dict{type_dict},
          mem_lock(shm_descriptor, object_memory - memory_start, obj_type.length) {
    }

    Field& get_field (std::string& name) {

        if (obj_type.field_map.count(name) == 0) {
            std::string msg = "store object has no attribute \"" + name + "\"";
            throw py::attribute_error(msg);
        }

        return obj_type[name];
    }

    void write_lock_field (std::string& name) {

        Field& field = get_field(name);

        RangeLock(shm_descriptor,
                  object_memory - memory_start + field.offset,
                  field.length).lock(F_WRLCK);
    }

    void read_lock_field (std::string& name) {

        Field& field = get_field(name);

        RangeLock(shm_descriptor,
                  object_memory - memory_start + field.offset,
                  field.length).lock(F_RDLCK);
    }

    void unlock_field (std::string& name) {

        Field& field = get_field(name);

        RangeLock(shm_descriptor,
                  object_memory - memory_start + field.offset,
                  field.length).unlock();
    }
    
    std::vector<std::string> slots () {

        std::vector<std::string> s;

        for (Field& f : obj_type.fields) {
            s.push_back(f.name);
        }

        return s;
    }

    StoreObject copy () {

        StoreObject so(-1, 0, 0, obj_type_qualname, obj_type, type_dict);

        so.own_data = std::make_shared<std::vector<uint8_t>>(obj_type.length);
        std::memcpy(so.own_data->data(), object_memory, obj_type.length);

        so.memory_start = so.own_data->data();
        so.object_memory = so.own_data->data();

        return so;
    }
};

struct Store {

    std::string store_name;
    int shm_descriptor = -1;
    size_t total_size = 0;
    uint64_t creation_time = 0;

    bool server = false;

    TypeDict type_dict;

    uint8_t* memory_start = nullptr;
    uint8_t* object_tree_start = nullptr;

    std::string main_obj_type_name;

    Store (std::string store_name) {

        if (store_name.find("/") == store_name.npos) {
            this->store_name = "/" + store_name;
        } else {
            throw py::value_error("store name must not contain \"/\"");
        }
    }

    void init_server (py::object& obj) {

        server = true;

        shm_descriptor = shm_open(this->store_name.c_str(), O_CREAT | O_RDWR, 0600);

        // if we find an opened memory segment with a non-zero size,
        // it's likely an old segment, thus we unlink and recreate it
        
        struct stat shm_stat;
        std::memset(&shm_stat, 0, sizeof(struct stat));
        fstat(shm_descriptor, &shm_stat);

        if (shm_stat.st_size != 0) {
            shm_unlink(store_name.c_str());
            close(shm_descriptor);

            shm_descriptor = shm_open(this->store_name.c_str(), O_CREAT | O_RDWR, 0600);
        }

        if (0 > shm_descriptor) {
            throw std::runtime_error("opening shared memory failed");
        }

        // creating type dict and resizing shared memory

        build_type_dict(type_dict, obj);

        main_obj_type_name = get_fq_type_name(obj);

        total_size = type_dict_size(type_dict)
                   + sizeof(uint32_t) + main_obj_type_name.length()
                   + type_dict[main_obj_type_name].length;

        if (0 > ftruncate(shm_descriptor, total_size)) {
            throw std::runtime_error("truncating shared memory failed");
        }

        memory_start = (uint8_t*)mmap(
                0, 
                total_size,
                PROT_READ | PROT_WRITE, 
                MAP_SHARED, 
                shm_descriptor,
                0);

        // writing initial data

        uint8_t* data_seg = write_type_dict(type_dict, memory_start);

        write_val(data_seg, main_obj_type_name);
        object_tree_start = data_seg;

        write_object(object_tree_start, type_dict, obj, main_obj_type_name);

        // marks the store as ready to be used by clients
        
        fchmod(shm_descriptor, 0666);
    }

    void init_client () {

        // wait for the server to finish initialization 
        
        struct stat s;
        std::memset(&s, 0, sizeof(struct stat));

        do {
            // get out of the loop, if necessary
            if (PyErr_CheckSignals() != 0) { 
                throw py::error_already_set();
            }
            
            // shameful busy waiting, ...
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (shm_descriptor == -1) {
                shm_descriptor = shm_open(this->store_name.c_str(), O_RDWR, 0600);
            }
            fstat(shm_descriptor, &s);
        } while(s.st_mode != 0100666); // ... it's the devil's business

        creation_time = s.st_ctim.tv_sec * 10e9 + s.st_ctim.tv_nsec;

        total_size = s.st_size;

        memory_start = (uint8_t*)mmap(
                0, 
                s.st_size,
                PROT_READ | PROT_WRITE, 
                MAP_SHARED, 
                shm_descriptor,
                0);

        type_dict.clear();

        object_tree_start = read_type_dict(type_dict, memory_start);
        read_val(object_tree_start, main_obj_type_name);
    }

    ~Store () {

        if (memory_start != nullptr) {
            munmap(memory_start, total_size);
        }
        if (shm_descriptor != -1 && server) {
            shm_unlink(store_name.c_str());
        }
    }

    void check_for_upgrade () {

        if (server) {
            return;
        }

        int tmp_descriptor = shm_open(this->store_name.c_str(), O_RDWR, 0600);

        if (tmp_descriptor == -1) {
            return;
        }

        struct stat s;
        if (0 > fstat(shm_descriptor, &s)) {
            return;
        }

        close(tmp_descriptor);

        size_t new_time = s.st_ctim.tv_sec * 10e9 + s.st_ctim.tv_nsec;

        if (new_time <= creation_time) {
            return;
        }

        if (memory_start != nullptr) {
            munmap(memory_start, total_size);
        }
        close(shm_descriptor);
        shm_descriptor = -1;

        init_client();
    }

    void write_lock () {
        RangeLock(shm_descriptor, 0, total_size).lock(F_WRLCK);
    }

    void read_lock () {
        RangeLock(shm_descriptor, 0, total_size).lock(F_RDLCK);
    }

    void unlock () {
        RangeLock(shm_descriptor, 0, total_size).unlock();
    }

    StoreObject get () {

        return StoreObject(
            shm_descriptor,
            memory_start,
            object_tree_start,
            main_obj_type_name,
            type_dict[main_obj_type_name],
            type_dict);
    }
};

PYBIND11_MODULE(sharedstate, m) {

    py::class_<StoreObject>(m, "StoreObject")
        .def("__setattr__", [](StoreObject& self, std::string name, py::object value) {

            Field& field = self.get_field(name);
            write_field(self.object_memory, self.type_dict, field, value);
        })
        .def("__getattr__", [](StoreObject& self, std::string name) {

            Field& f = self.get_field(name);
            uint8_t* ptr = self.object_memory + f.offset;

            switch (f.kind) {
                case KIND_INT: {
                    int64_t value = 0;
                    read_val(ptr, value);
                    return py::object(py::int_(value));
                }
                case KIND_FLOAT: {
                    double value = 0;
                    read_val(ptr, value);
                    return py::object(py::float_(value));
                }
                case KIND_BOOL: {
                    uint8_t value = 0;
                    read_val(ptr, value);
                    return py::object(py::bool_(value));
                }
                case KIND_STRING: {
                    std::string value;
                    read_val(ptr, value);
                    return py::cast(value);
                }
                case KIND_ARRAY: {
                    ssize_t ndim = 0;
                    read_val(ptr, ndim);
                    std::vector<ssize_t> shape(ndim);
                    for (ssize_t k = 0; k < ndim; ++k) {
                        read_val(ptr, shape[k]);
                    }
                    return py::object(array_like<double>(shape, (double*)ptr));
                }
                case KIND_STRUCT: {
                    return py::cast(
                        StoreObject(
                            self.shm_descriptor,
                            self.memory_start,
                            ptr,
                            f.ref_type_name,
                            self.type_dict[f.ref_type_name],
                            self.type_dict
                        )
                    );
                }
                default:
                    return py::object(py::none());
            }
        })
        .def_property_readonly("__slots__", &StoreObject::slots)
        .def("copy", &StoreObject::copy)
        .def("__copy__", [](StoreObject& self) {
            return self.copy();
        })
        .def("__deepcopy__", [](StoreObject& self, py::object& memo) {
            return self.copy();
        })
        .def("write_lock", [](StoreObject& self) {
            self.mem_lock.lock(F_WRLCK);
        })
        .def("read_lock", [](StoreObject& self) {
            self.mem_lock.lock(F_RDLCK);
        })
        .def("unlock", [](StoreObject& self){
            self.mem_lock.unlock();
        })
        .def("write_lock_field", &StoreObject::write_lock_field)
        .def("read_lock_field", &StoreObject::read_lock_field)
        .def("unlock_field", &StoreObject::unlock_field);

    py::class_<Store>(m, "Store")
        .def("get", &Store::get)
        .def("check_for_upgrade", &Store::check_for_upgrade)
        .def("write_lock", &Store::write_lock)
        .def("read_lock", &Store::read_lock)
        .def("unlock", &Store::unlock);

    m.def("create", [&](std::string store_name, py::object& obj) {

        std::unique_ptr<Store> store = std::make_unique<Store>(store_name);
        store->init_server(obj);

        return store;
    });

    m.def("open", [&](std::string store_name) {

        std::unique_ptr<Store> store = std::make_unique<Store>(store_name);
        store->init_client();

        return store;
    });
}
