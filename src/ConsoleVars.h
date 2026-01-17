#pragma once

#include <StringHash.h>

#include <string>
#include <string_view>

class ConsoleVars {
public:
    static void drawWithImGui();
};

template<typename T>
class CVarBase {
protected:
    using Type = T;

    CVarBase(size_t arrayIndex) : arrayIndex_(arrayIndex) {}

    size_t arrayIndex_;
};

class CVarInt : public CVarBase<int> {
public:
    static CVarInt create(StringHash name, std::string_view description, int value);
    static CVarInt access(StringHash name);
    int get();
    void set(int value);

private:
    CVarInt(size_t arrayIndex) : CVarBase(arrayIndex) {}
};

class CVarFloat : public CVarBase<float> {
public:
    static CVarFloat create(StringHash name, std::string_view description, float value);
    static CVarFloat access(StringHash name);
    float get();
    void set(float value);

private:
    CVarFloat(size_t arrayIndex) : CVarBase(arrayIndex) {}
};

class CVarString : public CVarBase<std::string> {
public:
    static CVarString create(StringHash name, std::string_view description, std::string_view value);
    static CVarString access(StringHash name);
    const std::string& get();
    void set(std::string_view value);

private:
    CVarString(size_t arrayIndex) : CVarBase(arrayIndex) {}
};
