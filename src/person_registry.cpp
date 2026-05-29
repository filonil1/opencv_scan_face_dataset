#include "person_registry.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace fdet {

// ─────────────────────────────────────────────────────────────────────────────
// Tiny self-contained JSON helpers (no external deps)
// ─────────────────────────────────────────────────────────────────────────────
namespace json {

std::string escape(const std::string& s) {
    std::string o; o.reserve(s.size());
    for (char c : s) {
        if      (c == '"')  o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else                o += c;
    }
    return o;
}

// Extract string value for a key: "key": "value"
std::string get_str(const std::string& src, const std::string& key) {
    std::string pat = "\"" + key + "\"";
    auto pos = src.find(pat);
    if (pos == std::string::npos) return "";
    pos = src.find(':', pos + pat.size());
    if (pos == std::string::npos) return "";
    pos = src.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    auto end = src.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return src.substr(pos + 1, end - pos - 1);
}

// Extract int value for a key: "key": 42
int get_int(const std::string& src, const std::string& key, int def = 0) {
    std::string pat = "\"" + key + "\"";
    auto pos = src.find(pat);
    if (pos == std::string::npos) return def;
    pos = src.find(':', pos + pat.size());
    if (pos == std::string::npos) return def;
    while (++pos < src.size() && (src[pos]==' '||src[pos]=='\t')) {}
    size_t end = pos;
    while (end < src.size() && std::isdigit(src[end])) ++end;
    if (end == pos) return def;
    return std::stoi(src.substr(pos, end - pos));
}

} // namespace json

// ─────────────────────────────────────────────────────────────────────────────

PersonRegistry::PersonRegistry(const std::string& dataset_root)
    : dataset_root_(dataset_root)
{
    fs::create_directories(dataset_root_);
    registry_path_ = (fs::path(dataset_root_) / "registry.json").string();
    load();
}

std::string PersonRegistry::format_id(int n) const {
    std::ostringstream ss;
    ss << std::setfill('0') << std::setw(3) << n;
    return ss.str();
}

std::string PersonRegistry::now_str() const {
    auto t = std::chrono::system_clock::to_time_t(
                 std::chrono::system_clock::now());
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M");
    return ss.str();
}

// ── Registration ─────────────────────────────────────────────────────────────

std::string PersonRegistry::add_person(const std::string& name) {
    // Check duplicate name
    if (auto it = name_to_id_.find(name); it != name_to_id_.end())
        return it->second;

    std::string id = format_id(next_id_++);
    PersonInfo  p{id, name, now_str(), 0};
    by_id_[id]        = p;
    name_to_id_[name] = id;

    // Create folder immediately
    fs::create_directories(fs::path(dataset_root_) / folder_name(id));
    save();
    std::cout << "[Registry] Added person  " << id << "  \"" << name << "\"\n";
    return id;
}

std::string PersonRegistry::get_or_create(const std::string& name) {
    if (auto it = name_to_id_.find(name); it != name_to_id_.end())
        return it->second;
    return add_person(name);
}

// ── Lookups ───────────────────────────────────────────────────────────────────

std::optional<PersonInfo> PersonRegistry::find_by_name(const std::string& name) const {
    auto it = name_to_id_.find(name);
    if (it == name_to_id_.end()) return std::nullopt;
    auto it2 = by_id_.find(it->second);
    if (it2 == by_id_.end()) return std::nullopt;
    return it2->second;
}

std::optional<PersonInfo> PersonRegistry::find_by_id(const std::string& id) const {
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return std::nullopt;
    return it->second;
}

std::string PersonRegistry::folder_name(const std::string& id) const {
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return id;
    return id + "_" + it->second.name;
}

// ── Mutation ──────────────────────────────────────────────────────────────────

void PersonRegistry::increment_samples(const std::string& id, int n) {
    auto it = by_id_.find(id);
    if (it != by_id_.end()) {
        it->second.samples += n;
        save();
    }
}

bool PersonRegistry::remove_person(const std::string& id) {
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return false;
    name_to_id_.erase(it->second.name);
    by_id_.erase(it);
    save();
    return true;
}

// ── Listing ───────────────────────────────────────────────────────────────────

std::vector<PersonInfo> PersonRegistry::all_persons() const {
    std::vector<PersonInfo> v;
    for (auto& [id, p] : by_id_) v.push_back(p);
    std::sort(v.begin(), v.end(),
              [](const PersonInfo& a, const PersonInfo& b){ return a.id < b.id; });
    return v;
}

void PersonRegistry::print_table() const {
    auto all = all_persons();
    std::cout << "\n┌──────┬──────────────────────┬─────────────────┬─────────┐\n"
              << "│  ID  │  Name                │  Added          │ Samples │\n"
              << "├──────┼──────────────────────┼─────────────────┼─────────┤\n";
    for (auto& p : all) {
        std::cout << "│ " << std::left << std::setw(4) << p.id
                  << " │ " << std::setw(20) << p.name
                  << " │ " << std::setw(15) << p.added_at
                  << " │ " << std::right << std::setw(7) << p.samples
                  << " │\n";
    }
    std::cout << "└──────┴──────────────────────┴─────────────────┴─────────┘\n"
              << "  Total: " << all.size() << " person(s)\n\n";
}

// ── Persistence ───────────────────────────────────────────────────────────────

bool PersonRegistry::save() const {
    std::ofstream f(registry_path_);
    if (!f) return false;
    f << "{\n"
      << "  \"next_id\": " << next_id_ << ",\n"
      << "  \"persons\": {\n";
    bool first = true;
    for (auto& [id, p] : by_id_) {
        if (!first) f << ",\n";
        first = false;
        f << "    \"" << json::escape(id) << "\": {\n"
          << "      \"name\":    \"" << json::escape(p.name)     << "\",\n"
          << "      \"added\":   \"" << json::escape(p.added_at) << "\",\n"
          << "      \"samples\": " << p.samples << "\n"
          << "    }";
    }
    f << "\n  }\n}\n";
    return true;
}

bool PersonRegistry::load() {
    if (!fs::exists(registry_path_)) return true; // fresh start

    std::ifstream f(registry_path_);
    if (!f) return false;
    std::string src((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    next_id_ = json::get_int(src, "next_id", 1);

    // Parse each person block: find all "NNN": { … } inside "persons"
    auto persons_pos = src.find("\"persons\"");
    if (persons_pos == std::string::npos) return true;

    size_t cur = src.find('{', persons_pos + 1);
    if (cur == std::string::npos) return true;

    // Walk entries: "id": { ... }
    while (true) {
        // Find next quoted key (the id)
        auto q1 = src.find('"', cur + 1);
        if (q1 == std::string::npos) break;
        auto q2 = src.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        std::string id = src.substr(q1 + 1, q2 - q1 - 1);
        if (id == "persons" || id.empty()) { cur = q2 + 1; continue; }

        // Find the object for this id
        auto ob_s = src.find('{', q2 + 1);
        if (ob_s == std::string::npos) break;
        int depth = 0; size_t ob_e = ob_s;
        for (; ob_e < src.size(); ++ob_e) {
            if (src[ob_e]=='{') ++depth;
            else if (src[ob_e]=='}') { --depth; if (!depth) break; }
        }
        std::string obj = src.substr(ob_s, ob_e - ob_s + 1);

        PersonInfo p;
        p.id       = id;
        p.name     = json::get_str(obj, "name");
        p.added_at = json::get_str(obj, "added");
        p.samples  = json::get_int(obj, "samples");

        if (!p.name.empty()) {
            by_id_[id]          = p;
            name_to_id_[p.name] = id;
        }
        cur = ob_e + 1;
        if (cur >= src.size()) break;
    }
    return true;
}

} // namespace fdet
