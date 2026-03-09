#include <Engine.h>

#include <spdlog/spdlog.h>

#include <Scene.h>
#include <SceneView.h>
#include <components/Transform.h>
#include <iostream>
#include <GlmFormatters.h>
#include <glm/gtc/type_ptr.hpp>
#include <components/Camera.h>
#include <components/Renderable.h>
#include <Loader.h>

#include <SDL3/SDL.h>

struct EvilComp {
    EvilComp(EntityId* target) {
        this->target = target;
        myData = new std::string(3, nextData++);
        std::cout << "EvilComp::ctor(), this = " << this << "\n";
        std::cout << "  myData = " << *myData << "\n";
    }
    ~EvilComp() {
        std::cout << "EvilComp::dtor(), this = " << this << "\n";
        std::cout << "  myData = " << (myData != nullptr ? *myData : "null") << "\n";
        delete myData;
        auto& scene = Scene::instance();
        if (target != nullptr) {
            std::cout << "target " << *target << " will be destroyed\n";
            scene.destroyEntity(*target);
        }

        std::cout << "EvilComp::dtor() finished, this = " << this << "\n";
    }
    EvilComp& operator=(EvilComp&& rhs) noexcept {
        std::cout << "EvilComp::move assign()\n";
        std::cout << "  myData = " << *myData << ", rhs = " << *rhs.myData << "\n";
        auto& scene = Scene::instance();
        if (target != nullptr) {
            std::cout << "target " << *target << " will be destroyed\n";
            scene.destroyEntity(*target);
        }
        target = std::move(rhs.target);
        rhs.target = nullptr;
        myData = std::move(rhs.myData);
        rhs.myData = nullptr;
        return *this;
    }
    EntityId* target;
    std::string* myData;
    static char nextData;
};
char EvilComp::nextData = 'A';

int main() {
    spdlog::set_level(spdlog::level::debug);
    spdlog::info("Using spdlog v{}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);
    spdlog::info("Logging level set to {}.", spdlog::level::to_string_view(spdlog::get_level()));

    /*auto& s = Scene::instance();

    auto e1 = s.createEntity();
    auto t1 = Transform::addToScene(e1);

    auto e2 = s.createEntity();
    auto t2 = Transform::addToScene(e2, e1);

    t1->setPosition({2.0, 3.0, 4.0});

    spdlog::info("t1 pos = {}", t1->getPosition());
    spdlog::info("t1 rot = {}", t1->getOrientation());
    spdlog::info("t1 sca = {}", t1->getScale());

    spdlog::info("t1 = {}", glm::transpose(t1->getLocalTransform()));

    spdlog::info("t2 = {}", glm::transpose(t2->getLocalTransform()));

    spdlog::info("t1 = {}", glm::transpose(t1->getLocalTransform()));

    spdlog::info("t2 world = {}", glm::transpose(t2->getWorldTransform()));

    spdlog::info("t1 world = {}", glm::transpose(t1->getWorldTransform()));

    spdlog::info("scale t1 ##############################################");
    t1->setScale(glm::vec3{3.0});

    spdlog::info("t2 world = {}", glm::transpose(t2->getWorldTransform()));

    spdlog::info("t1 world = {}", glm::transpose(t1->getWorldTransform()));

    spdlog::info("manually destroy e2 ##############################");
    s.destroyEntity(e2);

    t1 = s.accessComponent<Transform>(e1);
    t1->setOrientation({0.5, 0.0, 0.5, 0.0});

    spdlog::info("t1 world = {}", glm::transpose(t1->getWorldTransform()));

    auto e3 = s.createEntity();
    auto t3 = Transform::addToScene(e3, e1);

    spdlog::info("manually destroy all");
    s.destroyAllEntities();

    spdlog::info("Finished running.");
    return 0;*/

    auto& scene = Scene::instance();
    auto ent0 = scene.createEntity();
    scene.assign<EvilComp>(ent0, nullptr);
    auto ent1 = scene.createEntity();
    scene.assign<EvilComp>(ent1, &ent0);
    scene.destroyEntity(ent1);

    std::cout << "Finished running (it should have crashed though)\n";
    return 0;

    {
        auto& s = Scene::instance();

        Engine engine;

        auto e0 = s.createEntity();
        auto t0 = components::Transform::addToScene(e0);

        t0->move({0.0f, 2.0f, 0.0f});

        auto e1 = s.createEntity();
        auto t1 = components::Transform::addToScene(e1, e0);

        auto cam = s.createEntity();
        auto camComp = components::Camera::addToScene(cam);
        auto camTrans = components::Transform::addToScene(cam);    // FIXME: would it be better to have Camera component do this instead? maybe not.

        engine.init();

        std::vector<std::shared_ptr<MeshAsset>> testMeshes;
        testMeshes = loadGltfMeshes(&engine, "assets/basicmesh.glb").value();

        components::Renderable::addToScene(e1, *testMeshes[2]);

        std::cout << "scene has " << s.getEntitiesCount() << " entities\n";
        for (auto entity : SceneView<components::Transform>(s)) {
            s.access<components::Transform>(entity)->printDebug(entity);
        }

        SDL_Event event;
        bool closeWindow = false;
        while (!closeWindow) {
            // Handle events from the queue.
            while (SDL_PollEvent(&event)) {
                engine.processEvent(&event);

                if (event.type == SDL_EVENT_QUIT) {
                    closeWindow = true;
                }
                if (event.type == SDL_EVENT_KEY_DOWN) {
                    if (event.key.key == SDLK_A) {
                        std::cout << "delete root node\n";
                        s.access<components::Transform>(e0)->printDebug(e0);
                        s.destroyEntity(e0);

                        std::cout << "scene has " << s.getEntitiesCount() << " entities\n";
                        for (auto entity : SceneView<components::Transform>(s)) {
                            s.access<components::Transform>(entity)->printDebug(entity);
                        }
                    }
                }
            }

            //t1->rotate(glm::vec3{0.0f, glm::radians(1.0f), 0.0f});

            engine.render();
        }

        engine.getDevice().waitIdle();
        for (auto& mesh : testMeshes) {
            mesh->meshBuffers.indexBuffer.clear(engine.getAllocator());
            mesh->meshBuffers.vertexBuffer.clear(engine.getAllocator());
        }

        engine.cleanup();
    }

    spdlog::info("Finished running.");
    return 0;
}
