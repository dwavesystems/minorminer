// Copyright 2022-2023 D-Wave Systems Inc.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
// Excerpted from and/or inspired by implementations found in the Boost Library
// see external/boost/BOOST_LICENSE or https://www.boost.org/LICENSE_1_0.txt

#pragma once

#include <condition_variable>
#include <mutex>
namespace boost {

class barrier {
    std::mutex mtx;
    std::condition_variable cv;
    const unsigned int waiters;
    unsigned int waiting;
    unsigned int generation;
  public:
    barrier(unsigned int m) : waiters(m), waiting(m), generation(0) {}
    bool wait() {
        std::unique_lock lk{mtx};
        unsigned int gen = generation;
        if (--waiting == 0) {
            generation++;
            waiting = waiters;
            lk.unlock();
            cv.notify_all();
            return true;
        }
        while (gen == generation) {
            cv.wait(lk);
        }
        return false;
    }
};

}
