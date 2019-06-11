Table of Contents
- [Reference](#reference)
- [Data Structures](#data-structures)
  - [General Tips](#general-tips)
  - [Linked List](#linked-list)
  - [Stack](#stack)
  - [Queue](#queue)
  - [Hash Table](#hash-table)
  - [Cache](#cache)
    - [LRU Cache](#lru-cache)
- [Algorithm](#algorithm)

## Reference
* [Leetcode](https://leetcode.com/)
  * [Classification](https://cspiration.com/leetcodeClassification#103)
    * Array
    * String
    * Math
    * Tree
    * Backtracking
    * Dynamic Programming
    * LinkedList
    * Binary Search
    * Matrix
    * DFS & BFS
    * Stack & Priority Queue
    * Bit Mamipulation
    * Topological Sort
    * Random
    * Graph
    * Union Find
    * Trie
    * Design
* [Codeforces](https://codeforces.com/)
  * If you can achieve **1800 scores within 1 hour** that would be highly recommended for Google applicants.
* [Interviewbit](https://www.interviewbit.com/courses/programming/)
  * Level1:
    * Time Complexity
  * Level2:
    * Arrays
    * Math
  * Level3:
    * Binary Search
    * Strings
    * Bit Manipulation
    * Two Pointers
  * Level4:
    * Linked Lists
    * Stack and Queues
  * Level5:
    * Backtracking
    * Hashing
  * Level6:
    * Heaps And Maps
    * Tree
  * Level7:
    * Dynamic Programming
    * Greedy Algorithm
  * Level8
    * Graph Data Structure & Algorithms
* [Cracking the coding interview](http://www.crackingthecodinginterview.com/)
  * Data Structures
    * Arrays and Strings
    * Linked List
    * Stacks and Queues
    * Tree and Graphs
  * Concepts and Algorithms
    * Bit Manipulation
    * Math and Logic Puzzles
    * Object-Oriented Design
    * System Design
    * Sorting and Searching
* Complexity
  * [bigocheatsheet](http://bigocheatsheet.com/)


## [Data Structures](https://github.com/kissofjudase23/Library-python-common-modules/tree/master/common/ds)

### General Tips
* The **"Stack"** can help for:
  * Palindrome issues
  * DFS
* The **"Queue"** can help for:
  * BFS

### Linked List
* Tips
  * The **"Runner"** Techinique (or second pointer)
    * The runner techinique means that you iterate through the linked list with two pointers simultaneously, with one head of the other.
    * Loop issues, find the middle, etc.

* [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/ds/linkedlist.py)
* Time complexity
  * Average and Worst

    |                              | Search | Push_front | Push_back | Remove Node |
    |------------------------------|--------|------------|-----------|-------------|
    | Singly Linked List with tail | O(n)   | O(1)       | O(1)      | O(n)        |
    | Doubly Linked List with tail | O(n)   | O(1)       | O(1)      | O(1)        |

### Stack
* [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/ds/stack.py)
* Time complexity
  * Average and Worst

      |       | Push | Pop  | Search|
      |-------|------|------|-------|
      | Stack | O(1) | O(1) | O(n)  |

### [Queue](./queue.py)
* [Implementation](https://github.com/kissofjudase23/Library-python-common-modules/blob/master/common/ds/queue.py)
* Time complexity
  * Average and Worst

    |       | Add  | Remove  | Search|
    |-------|------|---------|-------|
    | Queue | O(1) | O(1)    | O(n)  |


### Hash Table
* Time complexity
  * Average

    |            | Set  | Get  | Delete |
    |------------|------|------|--------|
    | Hash Table | O(1) | O(1) | O(1)   |

  * Worst

    |            | Set  | Get  | Delete |
    |------------|------|------|--------|
    | Hash Table | O(n) | O(n) | O(n)   |



### [Cache](./cache.py)
#### LRU Cache
* Time complexity
  * Doubly Linked List + Hash Table

    |            | Set  | Get  | Delete |
    |------------|------|------|--------|
    | LRU Cache  | O(1) | O(1) | O(1)   |

###Tree
* Time complexity
  * Average

    |                    | Access     | Search     | Insertion  | Deletion   |
    |--------------------|------------|------------|------------|------------|
    | Binary Search Tree | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |
    | AVL Tree           | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |
    | B Tree             | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |
    | Red-Black Tree     | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |

  * Worst

    |                    | Access     | Search     | Insertion  | Deletion   |
    |--------------------|------------|------------|------------|------------|
    | Binary Search Tree | O(n)       | O(n)       | O(n)       | O(n)       |
    | AVL Tree           | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |
    | B Tree             | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |
    | Red-Black Tree     | O(long(n)) | O(long(n)) | O(long(n)) | O(long(n)) |


## Algorithm
