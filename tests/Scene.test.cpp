#include <Scene.h>
#include <SceneView.h>

#include <algorithm>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <random>
#include <unordered_set>

// Wrapper types intended for the initial test only.
// Note that component types must be default constructible.
struct MyInt {
    MyInt() = default;
    MyInt(int v) : v(v) {}
    operator int() { return v; }
    int v;
};
struct MyDouble {
    MyDouble() = default;
    MyDouble(double v) : v(v) {}
    operator double() { return v; }
    double v;
};
struct MyString {
    MyString() = default;
    MyString(const std::string& v) : v(v) {}
    operator const std::string&() { return v; }
    std::string v;
};

TEST_CASE("Test initial", "[Scene]") {
    Scene s;

    // Catch2 does not specify the order these test cases will run, and in practice it seems random.
    // We cannot reset the value of `priv::componentIdCounter` because other component ids may have been decided already (and bound to static vars).
    auto componentIdCounterInitial = priv::componentIdCounter;

    // We should be able to iterate using component types that have not yet been instantiated.
    auto sceneView1 = SceneView<MyInt, MyDouble>(s);
    REQUIRE(sceneView1.begin() == sceneView1.end());

    REQUIRE(priv::componentIdCounter == componentIdCounterInitial + 2);

    auto e1 = s.createEntity();

    // Likewise, we should be able to access/remove one of these non-existent component types.
    REQUIRE(s.accessComponent<MyString>(e1) == nullptr);
    s.removeComponent<MyString>(e1);

    REQUIRE(priv::componentIdCounter == componentIdCounterInitial + 3);

    // Can assign a component for a new type.
    REQUIRE(s.assignComponent<MyString>(e1, "my string")->v == "my string");
    REQUIRE(s.accessComponent<MyString>(e1)->v == "my string");

    auto sceneView2 = SceneView<MyString>(s);
    auto sceneIter2 = sceneView2.begin();
    REQUIRE(*sceneIter2 == e1);
    ++sceneIter2;
    REQUIRE(sceneIter2 == sceneView2.end());

    auto sceneView3 = SceneView<MyString, MyDouble>(s);
    REQUIRE(sceneView3.begin() == sceneView3.end());

    auto e2 = s.createEntity();
    auto e3 = s.createEntity();
    REQUIRE(s.assignComponent<MyDouble>(e3, 1.3)->v == 1.3);
    REQUIRE(s.assignComponent<MyDouble>(e2, 1.2)->v == 1.2);

    // Iterating all entities returns them in order of creation.
    auto sceneView4 = SceneView<>(s);
    auto list1 = {e1, e2, e3};
    REQUIRE(std::equal(sceneView4.begin(), sceneView4.end(), list1.begin(), list1.end()));

    // Iterating specific component returns entities in order of the component assignment.
    auto sceneView5 = SceneView<MyDouble>(s);
    auto list2 = {e3, e2};
    REQUIRE(std::equal(sceneView5.begin(), sceneView5.end(), list2.begin(), list2.end()));
}

TEST_CASE("Test entity add/remove", "[Scene]") {
    Scene s;
    std::unordered_set<EntityId> seenEntities;

    auto e1 = s.createEntity();
    REQUIRE(seenEntities.insert(e1).second);
    auto e2 = s.createEntity();
    REQUIRE(seenEntities.insert(e2).second);
    auto e3 = s.createEntity();
    REQUIRE(seenEntities.insert(e3).second);

    {
        auto sceneView = SceneView<>(s);
        auto list = {e1, e2, e3};
        REQUIRE(std::equal(sceneView.begin(), sceneView.end(), list.begin(), list.end()));
    }

    s.destroyEntity(e1);
    {
        auto sceneView = SceneView<>(s);
        auto list = {e2, e3};
        REQUIRE(std::equal(sceneView.begin(), sceneView.end(), list.begin(), list.end()));
    }

    // Destroyed entity index can be reused with new id.
    auto e4 = s.createEntity();
    REQUIRE(seenEntities.insert(e4).second);
    {
        auto sceneView = SceneView<>(s);
        auto list = {e4, e2, e3};
        REQUIRE(std::equal(sceneView.begin(), sceneView.end(), list.begin(), list.end()));
    }

    auto e5 = s.createEntity();
    REQUIRE(seenEntities.insert(e5).second);
    {
        auto sceneView = SceneView<>(s);
        auto list = {e4, e2, e3, e5};
        REQUIRE(std::equal(sceneView.begin(), sceneView.end(), list.begin(), list.end()));
    }

    s.destroyEntity(e5);
    s.destroyEntity(e3);
    s.destroyEntity(e2);
    s.destroyEntity(e4);
    {
        auto sceneView = SceneView<>(s);
        REQUIRE(sceneView.begin() == sceneView.end());
    }

    auto e6 = s.createEntity();
    REQUIRE(seenEntities.insert(e6).second);
    auto e7 = s.createEntity();
    REQUIRE(seenEntities.insert(e7).second);
    auto e8 = s.createEntity();
    REQUIRE(seenEntities.insert(e8).second);
    {
        auto sceneView = SceneView<>(s);
        auto list = {e6, e7, e8};
        REQUIRE(std::equal(sceneView.begin(), sceneView.end(), list.begin(), list.end()));
    }
}

TEST_CASE("Test component add/remove", "[Scene]") {

}

struct Transform {
    double x, y, z;
};
struct Mesh {
    std::string meshName;
};
struct CharacterData {
    char data;
};

TEST_CASE("Test ECS example", "[Scene]") {
    Scene s;
    std::unordered_set<EntityId> seenEntities;

    std::vector<EntityId> rigidBodies, particles, npcs;
    size_t rigidBodiesDestroyed = 0;
    size_t particlesDestroyed = 0;
    size_t npcsDestroyed = 0;

    auto makeRigidBody = [&s,&seenEntities,&rigidBodies,&rigidBodiesDestroyed]() {
        auto e = s.createEntity();
        std::cout << "makeRigidBody(), e = " << e << "\n";
        REQUIRE(seenEntities.insert(e).second);

        double n = static_cast<double>(rigidBodies.size() + rigidBodiesDestroyed);
        s.assignComponent<Transform>(e, n + 0.0, n + 0.1, n + 0.2);
        s.assignComponent<Mesh>(e, "mesh" + std::to_string(rigidBodies.size() + rigidBodiesDestroyed));
        rigidBodies.push_back(e);
    };

    auto makeParticle = [&s,&seenEntities,&particles,&particlesDestroyed]() {
        auto e = s.createEntity();
        std::cout << "makeParticle(), e = " << e << "\n";
        REQUIRE(seenEntities.insert(e).second);

        double n = static_cast<double>(particles.size() + particlesDestroyed);
        s.assignComponent<Transform>(e, n + 0.3, n + 0.4, n + 0.5);
        particles.push_back(e);
    };

    auto makeNpc = [&s,&seenEntities,&npcs,&npcsDestroyed]() {
        auto e = s.createEntity();
        std::cout << "makeNpc(), e = " << e << "\n";
        REQUIRE(seenEntities.insert(e).second);

        double n = static_cast<double>(npcs.size() + npcsDestroyed);
        s.assignComponent<Transform>(e, n + 0.6, n + 0.7, n + 0.8);
        s.assignComponent<Mesh>(e, "mesh" + std::to_string(npcs.size() + npcsDestroyed));
        s.assignComponent<CharacterData>(e, 'c');
        npcs.push_back(e);
    };

    [[maybe_unused]] auto printEntityData = [&s]() {
        std::cout << "Entity data:\n";
        for (auto e : SceneView<>(s)) {
            std::cout << "entity " << e << "\t";
            auto t = s.accessComponent<Transform>(e);
            if (t != nullptr) {
                std::cout << " t={" << t->x << "," << t->y << "," << t->z << "}\t";
            } else {
                std::cout << " t=null\t";
            }

            auto m = s.accessComponent<Mesh>(e);
            if (m != nullptr) {
                std::cout << " m={" << m->meshName << "}\t";
            } else {
                std::cout << " m=null\t";
            }

            auto c = s.accessComponent<CharacterData>(e);
            if (c != nullptr) {
                std::cout << " c={" << c->data << "}\n";
            } else {
                std::cout << " c=null\n";
            }
        }
    };

    std::mt19937 mersenneRand(Catch::getSeed());
    std::uniform_real_distribution<> dist1(0.0, 1.0);

    for (size_t i = 0; i < 40; ++i) {
        double r = dist1(mersenneRand);
        if (r < 0.33333) {
            makeRigidBody();
        } else if (r < 0.66667) {
            makeParticle();
        } else {
            makeNpc();
        }
    }

    printEntityData();

    // Destroy some entities.
    for (size_t i = 0; i < 3 && !rigidBodies.empty(); ++i) {
        std::cout << "destroy rigidBody " << rigidBodies.front() << "\n";
        s.destroyEntity(rigidBodies.front());
        rigidBodies.erase(rigidBodies.begin());
        ++rigidBodiesDestroyed;
    }
    for (size_t i = 0; i < 2 && !particles.empty(); ++i) {
        std::cout << "destroy particle " << particles.front() << "\n";
        s.destroyEntity(particles.front());
        particles.erase(particles.begin());
        ++particlesDestroyed;
    }
    for (size_t i = 0; i < 4 && !npcs.empty(); ++i) {
        std::cout << "destroy npc " << npcs.front() << "\n";
        s.destroyEntity(npcs.front());
        npcs.erase(npcs.begin());
        ++npcsDestroyed;
    }
    size_t destroyCount = rigidBodiesDestroyed + particlesDestroyed + npcsDestroyed;

    printEntityData();

    // Create more to get back to the number we had.
    for (size_t i = 0; i < destroyCount; ++i) {
        double r = dist1(mersenneRand);
        if (r < 0.33333) {
            makeRigidBody();
        } else if (r < 0.66667) {
            makeParticle();
        } else {
            makeNpc();
        }
    }

    // One NPC is fancy.
    s.assignComponent<int>(npcs.back(), 123);

    printEntityData();

    // Update entities a few times.
    for (size_t i = 0; i < 3; ++i) {
        for (auto e : SceneView<Transform>(s)) {
            auto t = s.accessComponent<Transform>(e);
            t->x += 100.0;
            t->y += 100.0;
            t->z += 100.0;
        }

        for (auto e : SceneView<Mesh>(s)) {
            s.accessComponent<Mesh>(e)->meshName += "_";
        }

        for (auto e : SceneView<CharacterData, Mesh>(s)) {
            s.accessComponent<Mesh>(e)->meshName += "x";
        }

        // Update the fancy NPC.
        for (auto e : SceneView<int, CharacterData>(s)) {
            s.accessComponent<CharacterData>(e)->data += static_cast<char>(1);
        }
    }

    printEntityData();

    for (size_t i = 0; i < rigidBodies.size(); ++i) {
        size_t n = i + rigidBodiesDestroyed;
        auto t = s.accessComponent<Transform>(rigidBodies[i]);
        REQUIRE(t->x == n + 300.0);
        REQUIRE(t->y == n + 300.1);
        REQUIRE(t->z == n + 300.2);

        auto m = s.accessComponent<Mesh>(rigidBodies[i]);
        REQUIRE(m->meshName == "mesh" + std::to_string(n) + "___");

        auto c = s.accessComponent<CharacterData>(rigidBodies[i]);
        REQUIRE(c == nullptr);
    }

    for (size_t i = 0; i < particles.size(); ++i) {
        size_t n = i + particlesDestroyed;
        auto t = s.accessComponent<Transform>(particles[i]);
        REQUIRE(t->x == n + 300.3);
        REQUIRE(t->y == n + 300.4);
        REQUIRE(t->z == n + 300.5);

        auto m = s.accessComponent<Mesh>(particles[i]);
        REQUIRE(m == nullptr);

        auto c = s.accessComponent<CharacterData>(particles[i]);
        REQUIRE(c == nullptr);
    }

    for (size_t i = 0; i < npcs.size(); ++i) {
        size_t n = i + npcsDestroyed;
        auto t = s.accessComponent<Transform>(npcs[i]);
        REQUIRE(t->x == n + 300.6);
        REQUIRE(t->y == n + 300.7);
        REQUIRE(t->z == n + 300.8);

        auto m = s.accessComponent<Mesh>(npcs[i]);
        REQUIRE(m->meshName == "mesh" + std::to_string(n) + "_x_x_x");

        auto c = s.accessComponent<CharacterData>(npcs[i]);
        if (i + 1 < npcs.size()) {
            REQUIRE(c->data == 'c');
            REQUIRE(s.accessComponent<int>(npcs[i]) == nullptr);
        } else {
            REQUIRE(c->data == 'f');
            REQUIRE(*s.accessComponent<int>(npcs[i]) == 123);
        }
    }
}

// to test:
// entity id is unique: done
// entity add/remove: done
// component add/remove (removed component can change order)
// pointer/iterator validity is what I expect: meh
// lots of entities and components: wip