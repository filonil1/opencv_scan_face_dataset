#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  PersonRegistry — persistent ID ↔ Name mapping
//
//  Storage: dataset/registry.json
//  Format:
//  {
//    "next_id": 3,
//    "persons": {
//      "001": { "name": "Alice", "added": "2024-06-01 14:32", "samples": 42 },
//      "002": { "name": "Bob",   "added": "2024-06-01 15:10", "samples": 17 }
//    }
//  }
// ─────────────────────────────────────────────────────────────────────────────
#include <string>
#include <map>
#include <vector>
#include <optional>

namespace fdet {

struct PersonInfo {
    std::string id;        // zero-padded, e.g. "001"
    std::string name;      // human label, e.g. "Alice"
    std::string added_at;  // ISO-ish timestamp
    int         samples{0};// number of saved face crops
};

class PersonRegistry {
public:
    explicit PersonRegistry(const std::string& dataset_root = "dataset");

    // ── Registration ──────────────────────────────────────────────────────
    /// Register a new person. Returns their assigned ID.
    /// If the name already exists, returns the existing ID.
    std::string add_person(const std::string& name);

    /// Get or create: add if not found.
    std::string get_or_create(const std::string& name);

    // ── Lookups ───────────────────────────────────────────────────────────
    std::optional<PersonInfo> find_by_name(const std::string& name) const;
    std::optional<PersonInfo> find_by_id(const std::string& id)     const;

    /// Returns "ID_Name" folder name, e.g. "001_Alice"
    std::string folder_name(const std::string& id) const;

    // ── Mutation ──────────────────────────────────────────────────────────
    void increment_samples(const std::string& id, int n = 1);
    bool remove_person(const std::string& id);   // removes registry entry only

    // ── Listing ───────────────────────────────────────────────────────────
    std::vector<PersonInfo> all_persons() const;
    void print_table() const;

    // ── Persistence ───────────────────────────────────────────────────────
    bool load();
    bool save() const;

    std::string dataset_root() const { return dataset_root_; }

private:
    std::string format_id(int n) const;
    std::string now_str()        const;

    std::string                  dataset_root_;
    std::string                  registry_path_;
    int                          next_id_{1};
    std::map<std::string, PersonInfo> by_id_;   // id → info
    std::map<std::string, std::string> name_to_id_; // name → id
};

} // namespace fdet
