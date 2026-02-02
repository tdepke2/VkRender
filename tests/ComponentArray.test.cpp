#include <ComponentArray.h>

#include <algorithm>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

TEST_CASE("Test ctor", "[ComponentArray]") {
    ComponentArray<char> array1(10);
    REQUIRE(array1.size() == 0);
    REQUIRE(array1.capacity() == 10);
    REQUIRE(array1.begin() == array1.end());
}

TEST_CASE("Test access/modification", "[ComponentArray]") {
    ComponentArray<char> array1(10);
    REQUIRE_FALSE(array1.remove(3));
    REQUIRE_FALSE(array1.hasEntity(4));
    REQUIRE_THROWS_AS(array1.at(0), std::out_of_range);
    REQUIRE_THROWS_AS(array1.at(99), std::out_of_range);

    REQUIRE(*array1.assign(3, 'a') == 'a');
    REQUIRE(array1.at(3) == 'a');
    REQUIRE(array1[3] == 'a');
    REQUIRE(array1.hasEntity(3));
    REQUIRE(array1.size() == 1);
    REQUIRE(array1.capacity() == 10);
    REQUIRE(array1.begin() != array1.end());
    REQUIRE(*array1.assign(3, 'b') == 'b');

    REQUIRE(array1.remove(3));
    REQUIRE_THROWS_AS(array1.at(3), std::out_of_range);
    REQUIRE_FALSE(array1.hasEntity(3));
    REQUIRE(array1.size() == 0);
    REQUIRE(array1.capacity() == 10);
    REQUIRE(array1.begin() == array1.end());
    REQUIRE_FALSE(array1.remove(3));

    REQUIRE(*array1.assign(1, 'b') == 'b');
    REQUIRE(*array1.assign(2, 'c') == 'c');
    REQUIRE(*array1.assign(3, 'd') == 'd');
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

    REQUIRE(array1.remove(2));
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
    auto checkEquals = [](ComponentArray<std::string>& arr, std::vector<std::pair<uint32_t, std::string_view>> items) -> bool {
        auto arrIter = arr.begin();
        auto itemsIter = items.begin();
        size_t i = 0;
        while (arrIter != arr.end() && itemsIter != items.end()) {
            if (arrIter.getEntityIndex() != itemsIter->first || *arrIter != itemsIter->second) {
                std::cout << "checkEquals() found mismatch: arr[" << i << "] = {" << arrIter.getEntityIndex() << ", " << *arrIter << "}\n";
                return false;
            }
            ++arrIter;
            ++itemsIter;
            ++i;
        }
        if (arrIter != arr.end())  {
            std::cout << "checkEquals() found mismatch: arr has more elements, i = " << i << "\n";
            return false;
        }
        if (itemsIter != items.end())  {
            std::cout << "checkEquals() found mismatch: items has more elements, i = " << i << "\n";
            return false;
        }
        if (arr.size() != items.size()) {
            std::cout << "checkEquals() found mismatch: other checks passed but size didn\'t match.\n";
        }
        return true;
    };

    ComponentArray<std::string> arr1(10);
    REQUIRE(checkEquals(arr1, {}));
    arr1.assign(2, "second");
    REQUIRE(checkEquals(arr1, {
        {2, "second"},
    }));
    arr1.assign(3, "third");
    REQUIRE(checkEquals(arr1, {
        {2, "second"},
        {3, "third"},
    }));
    arr1.assign(5, "fifth");
    REQUIRE(checkEquals(arr1, {
        {2, "second"},
        {3, "third"},
        {5, "fifth"},
    }));

    // Removed element at end keeps order.
    arr1.remove(5);
    REQUIRE(checkEquals(arr1, {
        {2, "second"},
        {3, "third"},
    }));
    arr1.assign(5, "fifth");
    REQUIRE(checkEquals(arr1, {
        {2, "second"},
        {3, "third"},
        {5, "fifth"},
    }));

    // Removed element in middle gets replaced.
    arr1.remove(2);
    REQUIRE(checkEquals(arr1, {
        {5, "fifth"},
        {3, "third"},
    }));
    arr1.remove(5);
    REQUIRE(checkEquals(arr1, {
        {3, "third"},
    }));
    arr1.remove(3);
    REQUIRE(checkEquals(arr1, {}));

    arr1.assign(1, "a");
    arr1.assign(2, "b");
    arr1.assign(3, "c");
    arr1.assign(4, "d");
    arr1.assign(5, "e");
    REQUIRE(checkEquals(arr1, {
        {1, "a"},
        {2, "b"},
        {3, "c"},
        {4, "d"},
        {5, "e"},
    }));
    arr1.remove(1);
    arr1.remove(2);
    REQUIRE(checkEquals(arr1, {
        {5, "e"},
        {4, "d"},
        {3, "c"},
    }));
    arr1.assign(2, "b");
    arr1.assign(1, "a");
    REQUIRE(checkEquals(arr1, {
        {5, "e"},
        {4, "d"},
        {3, "c"},
        {2, "b"},
        {1, "a"},
    }));


    std::mt19937 mersenneRand(Catch::getSeed());
    std::uniform_real_distribution<> dist1(0.0, 1.0);

    ComponentArray<double> nums(100);
    std::vector<double> numsVec;

    for (uint32_t i = 0; i < 100; ++i) {
        double num = dist1(mersenneRand);
        auto result = nums.assign(i, num);
        REQUIRE(result.getEntityIndex() == i);
        REQUIRE(*result == num);

        numsVec.push_back(num);
    }

    REQUIRE(nums.size() == 100);
    REQUIRE(nums.capacity() == 100);
    REQUIRE(std::equal(nums.begin(), nums.end(), numsVec.begin()));

    numsVec.clear();
    for (uint32_t i = 0; i < 100; ++i) {
        // Repeatedly remove first item in container.
        REQUIRE(nums.remove(i));
    }

    REQUIRE(nums.size() == 0);
    REQUIRE(nums.capacity() == 100);
    REQUIRE(std::equal(nums.begin(), nums.end(), numsVec.begin()));
}
