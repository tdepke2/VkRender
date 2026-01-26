#include <ComponentArray.h>

#include <algorithm>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <stdexcept>
#include <vector>

TEST_CASE("Test ctor", "[ComponentArray]") {
    ComponentArray<char> array1(10);
    REQUIRE(array1.size() == 0);
    REQUIRE(array1.capacity() == 10);
    REQUIRE(array1.begin() == array1.end());


}

TEST_CASE("Test access/modification", "[ComponentArray]") {
    ComponentArray<char> array1(10);
    REQUIRE_FALSE(array1.erase(3));
    REQUIRE_FALSE(array1.hasEntity(3));
    REQUIRE_THROWS_AS(array1.at(0), std::out_of_range);
    REQUIRE_THROWS_AS(array1.at(99), std::out_of_range);

    REQUIRE(array1.emplace(3, 'a').second);
    REQUIRE(array1.at(3) == 'a');
    REQUIRE(array1[3] == 'a');
    REQUIRE(array1.hasEntity(3));
    REQUIRE(array1.size() == 1);
    REQUIRE(array1.capacity() == 10);
    REQUIRE(array1.begin() != array1.end());
    REQUIRE_FALSE(array1.emplace(3, 'a').second);

    REQUIRE(array1.erase(3));
    REQUIRE_THROWS_AS(array1.at(3), std::out_of_range);
    REQUIRE_FALSE(array1.hasEntity(3));
    REQUIRE(array1.size() == 0);
    REQUIRE(array1.capacity() == 10);
    REQUIRE(array1.begin() == array1.end());
    REQUIRE_FALSE(array1.erase(3));

    REQUIRE(array1.emplace(1, 'b').second);
    REQUIRE(array1.emplace(2, 'c').second);
    REQUIRE(array1.emplace(3, 'd').second);
    REQUIRE(array1.at(1) == 'b');
    REQUIRE(array1.at(2) == 'c');
    REQUIRE(array1.at(3) == 'd');
    REQUIRE_FALSE(array1.hasEntity(0));
    REQUIRE(array1.hasEntity(1));
    REQUIRE(array1.hasEntity(2));
    REQUIRE(array1.hasEntity(3));
    REQUIRE_FALSE(array1.hasEntity(4));
    REQUIRE(array1.size() == 3);
    REQUIRE(array1.capacity() == 10);
    REQUIRE(array1.begin() != array1.end());

    REQUIRE(array1.erase(2));
    REQUIRE(array1.at(1) == 'b');
    REQUIRE_THROWS_AS(array1.at(2), std::out_of_range);
    REQUIRE(array1.at(3) == 'd');
    REQUIRE_FALSE(array1.hasEntity(0));
    REQUIRE(array1.hasEntity(1));
    REQUIRE_FALSE(array1.hasEntity(2));
    REQUIRE(array1.hasEntity(3));
    REQUIRE_FALSE(array1.hasEntity(4));
    REQUIRE(array1.size() == 2);
    REQUIRE(array1.capacity() == 10);
    REQUIRE(array1.begin() != array1.end());


    
}

TEST_CASE("Test iterators", "[ComponentArray]") {
    std::mt19937 mersenneRand(Catch::getSeed());
    std::uniform_real_distribution<> dist1(0.0, 1.0);

    ComponentArray<double> nums(100);
    std::vector<double> numsVec;

    for (uint32_t i = 0; i < 100; ++i) {
        double num = dist1(mersenneRand);
        auto result = nums.emplace(i, num);
        REQUIRE(result.first.getEntityIndex() == i);
        REQUIRE(*result.first == num);
        REQUIRE(result.second);

        numsVec.push_back(num);
    }

    REQUIRE(nums.size() == 100);
    REQUIRE(nums.capacity() == 100);
    REQUIRE(std::equal(nums.begin(), nums.end(), numsVec.begin()));

    numsVec.clear();
    for (uint32_t i = 0; i < 100; ++i) {
        // Repeatedly erase first item in container.
        REQUIRE(nums.erase(i));
    }

    REQUIRE(nums.size() == 0);
    REQUIRE(nums.capacity() == 100);
    REQUIRE(std::equal(nums.begin(), nums.end(), numsVec.begin()));


}
