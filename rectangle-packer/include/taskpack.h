/*
MIT License

Copyright (c) 2017 Daniel Andersson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#ifndef TASKPACK_H
#define TASKPACK_H

#include <stdlib.h>

typedef struct dataunit {
    double duration;
    size_t group; /* Special value: (size_t)-1 <=> belongs to no group*/
    size_t id;
} Task;

/*
 * Function:  taskpack_algorithm
 * --------------------
 * Given a collection of tasks with known durations and given a set of
 * groups: Divide the tasks into the different groups in such a way,
 * that the total duration of a group is minimized.
 *
 *  tasks: An array of Task structs. A Task has a duration, assigned
 *         group and an id.
 *  nr_tasks: The number of tasks in `tasks` array.
 *  nr_groups: The number of groups available.
 *
 *  returns: Status. Error is indicated by -1.
 */
int taskpack_algorithm(Task* tasks, size_t nr_tasks, size_t nr_groups);

#endif
