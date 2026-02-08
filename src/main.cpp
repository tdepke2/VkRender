#include <Engine.h>

#include <spdlog/spdlog.h>

#include <Scene.h>
#include <Transform.h>

int main() {
    spdlog::set_level(spdlog::level::debug);
    spdlog::info("Using spdlog v{}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);
    spdlog::info("Logging level set to {}.", spdlog::level::to_string_view(spdlog::get_level()));

    auto& s = Scene::instance();

    auto e1 = s.createEntity();
    auto t1 = Transform::addToScene(e1);

    auto e2 = s.createEntity();
    auto t2 = Transform::addToScene(e2, e1);

    return 0;

    {
        Engine engine;

        engine.init();
        engine.run();
        engine.cleanup();
    }

    spdlog::info("Finished running.");
    return 0;
}
