#include <ConsoleVars.h>

#include <algorithm>
#include <spdlog/fmt/fmt.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

enum class CVarType {
    INT,
    FLOAT,
    STRING
};

constexpr std::string_view getCVarTypeString(CVarType type) {
    switch(type) {
    case CVarType::INT:
        return "int";
    case CVarType::FLOAT:
        return "float";
    case CVarType::STRING:
        return "string";
    default:
        return "???";
    }
}

class ConsoleVarsImpl {
public:
    struct CVarDetails {
        size_t arrayIndex;
        CVarType type;
        std::string name;
        std::string description;
    };

    template<typename T>
    struct CVarStorage {
        T initial;
        T current;
    };

    static ConsoleVarsImpl* instance();
    virtual ~ConsoleVarsImpl() = default;
    ConsoleVarsImpl(const ConsoleVarsImpl& rhs) = delete;
    ConsoleVarsImpl& operator=(const ConsoleVarsImpl& rhs) = delete;

    void addLookup(StringHash name, std::string_view description, CVarType type, size_t arrayIndex);
    template<typename T>
    size_t add(StringHash name, std::string_view description, T value);
    size_t find(StringHash name, CVarType type);
    template<typename T>
    CVarStorage<T>& accessStorage(size_t arrayIndex);

    void drawWithImGui();

private:
    ConsoleVarsImpl() = default;

    std::unordered_map<uint32_t, CVarDetails> lookupCVars_;
    std::vector<CVarDetails*> sortedCVars_;

    // These could be arrays instead, but I doubt there will be thousands of cvars in use that would make the static memory pay off.
    std::vector<CVarStorage<int>> intCVars_;
    std::vector<CVarStorage<float>> floatCVars_;
    std::vector<CVarStorage<std::string>> stringCVars_;
};

ConsoleVarsImpl* ConsoleVarsImpl::instance() {
    static ConsoleVarsImpl instance;
    return &instance;
}

void ConsoleVarsImpl::addLookup(StringHash name, std::string_view description, CVarType type, size_t arrayIndex) {
    auto existingCVar = lookupCVars_.find(name.hash);
    if (existingCVar != lookupCVars_.end()) {
        throw std::runtime_error(fmt::format("failed to add cvar \"{}\", hash collision with existing cvar \"{}\" (value {}).", name.str, existingCVar->second.name, existingCVar->first));
    }

    lookupCVars_.emplace(name.hash, CVarDetails{
        .arrayIndex = arrayIndex,
        .type = type,
        .name = static_cast<std::string>(name.str),
        .description = static_cast<std::string>(description)
    });
}

template<>
size_t ConsoleVarsImpl::add(StringHash name, std::string_view description, int value) {
    addLookup(name, description, CVarType::INT, intCVars_.size());
    intCVars_.push_back({
        .initial = value,
        .current = value
    });
    return intCVars_.size() - 1;
}

template<>
size_t ConsoleVarsImpl::add(StringHash name, std::string_view description, float value) {
    addLookup(name, description, CVarType::FLOAT, floatCVars_.size());
    floatCVars_.push_back({
        .initial = value,
        .current = value
    });
    return floatCVars_.size() - 1;
}

template<>
size_t ConsoleVarsImpl::add(StringHash name, std::string_view description, std::string_view value) {
    addLookup(name, description, CVarType::STRING, stringCVars_.size());
    stringCVars_.push_back({
        .initial = static_cast<std::string>(value),
        .current = static_cast<std::string>(value)
    });
    return stringCVars_.size() - 1;
}

size_t ConsoleVarsImpl::find(StringHash name, CVarType type) {
    auto cvar = lookupCVars_.find(name.hash);
    if (cvar == lookupCVars_.end() || cvar->second.type != type) {
        throw std::runtime_error(fmt::format("failed to find cvar \"{}\" of type {}.", name.str, getCVarTypeString(type)));
    }

    return cvar->second.arrayIndex;
}

template<>
ConsoleVarsImpl::CVarStorage<int>& ConsoleVarsImpl::accessStorage(size_t arrayIndex) {
    return intCVars_[arrayIndex];
}

template<>
ConsoleVarsImpl::CVarStorage<float>& ConsoleVarsImpl::accessStorage(size_t arrayIndex) {
    return floatCVars_[arrayIndex];
}

template<>
ConsoleVarsImpl::CVarStorage<std::string>& ConsoleVarsImpl::accessStorage(size_t arrayIndex) {
    return stringCVars_[arrayIndex];
}

void ConsoleVarsImpl::drawWithImGui() {
    sortedCVars_.clear();
    for (auto& cvar : lookupCVars_) {
        sortedCVars_.push_back(&cvar.second);
    }

    std::sort(sortedCVars_.begin(), sortedCVars_.end(), [](CVarDetails* a, CVarDetails* b) {
        return a->name < b->name;
    });

    // Organize the cvars by category (pattern up to the first '.' in the string).
    // FIXME: going to skip this for now, but https://vkguide.dev/docs/extra-chapter/cvar_system/ has a good reference implementation.
    /*std::string lastCategory = "";
    for (const auto& cvar : sortedCVars_) {
        size_t firstDot = cvar->name.find('.');
        if (firstDot != std::string::npos && firstDot > 0) {
            std::string category = cvar->name.substr(0, firstDot);
            if (category != lastCategory) {

            }
        } else {

        }
    }*/

    for (const auto& cvar : sortedCVars_) {
        if (cvar->type == CVarType::INT) {
            ImGui::InputInt(cvar->name.c_str(), &intCVars_[cvar->arrayIndex].current);
        } else if (cvar->type == CVarType::FLOAT) {
            ImGui::InputFloat(cvar->name.c_str(), &floatCVars_[cvar->arrayIndex].current, 0, 0, "%.3f");
        } else if (cvar->type == CVarType::STRING) {
            ImGui::InputText(cvar->name.c_str(), &stringCVars_[cvar->arrayIndex].current);
        }

        if (ImGui::IsItemHovered() && !cvar->description.empty()) {
            ImGui::SetTooltip("%s", cvar->description.c_str());
        }
    }
}

void ConsoleVars::drawWithImGui() {
    ConsoleVarsImpl::instance()->drawWithImGui();
}


CVarInt CVarInt::create(StringHash name, std::string_view description, int value) {
    return {ConsoleVarsImpl::instance()->add(name, description, value)};
}

CVarInt CVarInt::access(StringHash name) {
    return {ConsoleVarsImpl::instance()->find(name, CVarType::INT)};
}

int CVarInt::get() {
    return ConsoleVarsImpl::instance()->accessStorage<Type>(arrayIndex_).current;
}

void CVarInt::set(int value) {
    ConsoleVarsImpl::instance()->accessStorage<Type>(arrayIndex_).current = value;
}


CVarFloat CVarFloat::create(StringHash name, std::string_view description, float value) {
    return {ConsoleVarsImpl::instance()->add(name, description, value)};
}

CVarFloat CVarFloat::access(StringHash name) {
    return {ConsoleVarsImpl::instance()->find(name, CVarType::FLOAT)};
}

float CVarFloat::get() {
    return ConsoleVarsImpl::instance()->accessStorage<Type>(arrayIndex_).current;
}

void CVarFloat::set(float value) {
    ConsoleVarsImpl::instance()->accessStorage<Type>(arrayIndex_).current = value;
}


CVarString CVarString::create(StringHash name, std::string_view description, std::string_view value) {
    return {ConsoleVarsImpl::instance()->add(name, description, value)};
}

CVarString CVarString::access(StringHash name) {
    return {ConsoleVarsImpl::instance()->find(name, CVarType::STRING)};
}

const std::string& CVarString::get() {
    return ConsoleVarsImpl::instance()->accessStorage<Type>(arrayIndex_).current;
}

void CVarString::set(std::string_view value) {
    ConsoleVarsImpl::instance()->accessStorage<Type>(arrayIndex_).current = value;
}
