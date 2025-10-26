#include <Test.h>

#include <iostream>
#include <spdlog/spdlog.h>

int main() {
    spdlog::set_level(spdlog::level::info);
    Test t;
    return 0;
}
