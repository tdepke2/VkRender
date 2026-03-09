#include <ComponentArray.h>
#include <DisableWarning.h>

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
    ComponentArray<char> array1;
    REQUIRE(array1.size() == 0);
    REQUIRE(array1.begin() == array1.end());
}

TEST_CASE("Test access/modification", "[ComponentArray]") {
    ComponentArray<char> array1;
    REQUIRE_FALSE(array1.remove(3));
    REQUIRE_FALSE(array1.hasEntity(4));
    REQUIRE_THROWS_AS(array1.at(0), std::out_of_range);
    REQUIRE_THROWS_AS(array1.at(99), std::out_of_range);

    REQUIRE(*array1.assign(3, 'a') == 'a');
    REQUIRE(array1.at(3) == 'a');
    REQUIRE(array1[3] == 'a');
    REQUIRE(array1.hasEntity(3));
    REQUIRE(array1.size() == 1);
    REQUIRE(array1.begin() != array1.end());
    REQUIRE(*array1.assign(3, 'b') == 'b');

    REQUIRE(array1.remove(3));
    REQUIRE_THROWS_AS(array1.at(3), std::out_of_range);
    REQUIRE_FALSE(array1.hasEntity(3));
    REQUIRE(array1.size() == 0);
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
    REQUIRE(array1.begin() != array1.end());
}

// Track some data and a counter for the number of "live" data elements.
struct TrackedData {
    TrackedData() {
        std::cout << "TrackedData::TrackedData() default ctor\n";
    }
    TrackedData(int val, size_t& counter) :
        val(val),
        counter(&counter) {
        ++(*this->counter);
        std::cout << "TrackedData::TrackedData() value ctor, val = " << val << ", counter = " << *this->counter << "\n";
    }
    ~TrackedData() {
        if (counter != nullptr) {
            --*counter;
        }
        std::cout << "TrackedData::~TrackedData() dtor, val = " << val << ", counter = " << (counter == nullptr ? "nullptr" : std::to_string(*counter)) << "\n";
    }
    TrackedData(const TrackedData& rhs) = delete;
    TrackedData(TrackedData&& rhs) noexcept :
        val(std::move(rhs.val)),
        counter(std::move(rhs.counter)) {
        rhs.counter = nullptr;
        std::cout << "TrackedData::TrackedData() move ctor, val = " << val << "\n";
    }
    TrackedData& operator=(const TrackedData& rhs) = delete;
    TrackedData& operator=(TrackedData&& rhs) noexcept {
        if (counter != nullptr) {
            --*counter;
        }
        val = std::move(rhs.val);
        counter = std::move(rhs.counter);
        rhs.counter = nullptr;
        std::cout << "TrackedData::operator=() move assign, val = " << val << ", counter = " << (counter == nullptr ? "nullptr" : std::to_string(*counter)) << "\n";
        return *this;
    }

    int val = 0;
    size_t* counter = nullptr;
};

TEST_CASE("Test move/destruction of components", "[ComponentArray]") {
    size_t counter = 0;

    // Removing a component destroys it.
    std::cout << "Removing a component destroys it.\n";
    {
        ComponentArray<TrackedData> array1;

        REQUIRE(array1.assign(9, 123, counter)->val == 123);
        REQUIRE(counter == 1);
        REQUIRE(array1.assign(9, 456, counter)->val == 456);
        REQUIRE(counter == 1);
        REQUIRE(array1.remove(9));
        REQUIRE(counter == 0);
    }
    REQUIRE(counter == 0);

    // Destructor destroys all components.
    std::cout << "Destructor destroys all components.\n";
    {
        ComponentArray<TrackedData> array1;

        REQUIRE(array1.assign(9, 123, counter)->val == 123);
        REQUIRE(counter == 1);
        REQUIRE(array1.assign(8, 456, counter)->val == 456);
        REQUIRE(counter == 2);
    }
    REQUIRE(counter == 0);

    // Replacing a component with another destroys the old one.
    std::cout << "Replacing a component with another destroys the old one.\n";
    {
        ComponentArray<TrackedData> array1;

        REQUIRE(array1.assign(3, 12, counter)->val == 12);
        REQUIRE(counter == 1);
        REQUIRE(array1.assign(5, 34, counter)->val == 34);
        REQUIRE(counter == 2);
        REQUIRE(array1.remove(3));
        REQUIRE(counter == 1);
        REQUIRE(array1.assign(5, 56, counter)->val == 56);
        REQUIRE(counter == 1);
        REQUIRE(array1.assign(3, 78, counter)->val == 78);
        REQUIRE(counter == 2);
    }
    REQUIRE(counter == 0);
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

    ComponentArray<std::string> arr1;
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

    ComponentArray<double> nums;
    std::vector<double> numsVec;

    for (uint32_t i = 0; i < 100; ++i) {
        double num = dist1(mersenneRand);
        auto result = nums.assign(i, num);
        REQUIRE(result.getEntityIndex() == i);
        REQUIRE(*result == num);

        numsVec.push_back(num);
    }

    REQUIRE(nums.size() == 100);
    REQUIRE(std::equal(nums.begin(), nums.end(), numsVec.begin()));

    numsVec.clear();
    for (uint32_t i = 0; i < 100; ++i) {
        // Repeatedly remove first item in container.
        REQUIRE(nums.remove(i));
    }

    REQUIRE(nums.size() == 0);
    REQUIRE(std::equal(nums.begin(), nums.end(), numsVec.begin()));
}

struct DefaultAlign {
    int x;
};

DISABLE_WARNING_PUSH
DISABLE_WARNING_PADDED_DUE_TO_ALIGNMENT
struct alignas(256) CustomAlign {
    // With this memory alignment, the least significant byte of the address will be zero.
    int x;
};
DISABLE_WARNING_POP

TEST_CASE("Test alignment", "[ComponentArray]") {
    // We need to ensure that the component data respects memory alignment since
    // the internal implementation uses placement new to allocate objects into a
    // byte array. If memory alignment is not being followed, it could cause
    // inefficient access or segmentation faults on some architectures.
    {
        ComponentArray<DefaultAlign> arr1;

        arr1.assign(0, DefaultAlign{ .x = 7 });
        arr1.assign(3, DefaultAlign{ .x = 8 });
        std::cout << "DefaultAlign address is: " << &arr1.at(0) << "\n";
        REQUIRE(reinterpret_cast<std::uintptr_t>(&arr1.at(0)) % alignof(DefaultAlign) == 0);
        REQUIRE(reinterpret_cast<std::uintptr_t>(&arr1.at(3)) % alignof(DefaultAlign) == 0);
        REQUIRE(reinterpret_cast<std::uintptr_t>(&arr1.at(3)) == reinterpret_cast<std::uintptr_t>(&arr1.at(0)) + sizeof(DefaultAlign));
    }

    {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_PADDED_DUE_TO_ALIGNMENT
        ComponentArray<CustomAlign> arr1;
        DISABLE_WARNING_POP

        arr1.assign(0, CustomAlign{ .x = 9 });
        arr1.assign(3, CustomAlign{ .x = 10 });
        std::cout << "CustomAlign address is: " << &arr1.at(0) << "\n";
        REQUIRE(reinterpret_cast<std::uintptr_t>(&arr1.at(0)) % alignof(CustomAlign) == 0);
        REQUIRE(reinterpret_cast<std::uintptr_t>(&arr1.at(3)) % alignof(CustomAlign) == 0);
        REQUIRE(reinterpret_cast<std::uintptr_t>(&arr1.at(3)) == reinterpret_cast<std::uintptr_t>(&arr1.at(0)) + sizeof(CustomAlign));
    }
}
