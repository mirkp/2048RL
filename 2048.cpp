#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
using namespace std;

static const int SIZE = 4;
static const double INF = 1e18;
using Board = array<array<int,4>,4>;

// ===========================
// Board helpers
// ===========================
using Board = array<array<int,4>,4>;

int count_empty(const Board& b) {
    int c = 0;
    for (auto& r : b)
        for (int x : r)
            if (x == 0) c++;
    return c;
}

int max_tile(const Board& b) {
    int m = 0;
    for (auto& r : b)
        for (int x : r)
            m = max(m, x);
    return m;
}

// ===========================
// Move logic
// ===========================
pair<array<int,4>, int> slide_and_merge(array<int,4> row) {
    vector<int> v;
    for (int x : row)
        if (x) v.push_back(x);

    array<int,4> out = {0,0,0,0};
    int reward = 0;
    int idx = 0;

    for (size_t i = 0; i < v.size(); i++) {
        if (i + 1 < v.size() && v[i] == v[i + 1]) {
            out[idx] = v[i] * 2;
            reward += out[idx];
            i++;
        } else {
            out[idx] = v[i];
        }
        idx++;
    }
    return {out, reward};
}

bool move_board(const Board& b, Board& out, int action, int& reward) {
    out = b;
    reward = 0;

    for (int i = 0; i < 4; i++) {
        if (action == 0) { // UP
            array<int,4> col;
            for (int j = 0; j < 4; j++) col[j] = b[j][i];
            auto [m, r] = slide_and_merge(col);
            for (int j = 0; j < 4; j++) out[j][i] = m[j];
            reward += r;
        }
        else if (action == 2) { // DOWN
            array<int,4> col;
            for (int j = 0; j < 4; j++) col[j] = b[3 - j][i];
            auto [m, r] = slide_and_merge(col);
            for (int j = 0; j < 4; j++) out[3 - j][i] = m[j];
            reward += r;
        }
        else if (action == 3) { // LEFT
            auto [m, r] = slide_and_merge(b[i]);
            out[i] = m;
            reward += r;
        }
        else if (action == 1) { // RIGHT
            array<int,4> row;
            for (int j = 0; j < 4; j++) row[j] = b[i][3 - j];
            auto [m, r] = slide_and_merge(row);
            for (int j = 0; j < 4; j++) out[i][3 - j] = m[j];
            reward += r;
        }
    }
    return out != b;
}

// ===========================
// Heuristic
// ===========================
double evaluate(const Board& b) {
    int empty = count_empty(b);
    int maxT = max_tile(b);

    double smooth = 0;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (b[i][j]) {
                double v = log2(b[i][j]);
                if (i + 1 < 4 && b[i + 1][j])
                    smooth -= abs(v - log2(b[i + 1][j]));
                if (j + 1 < 4 && b[i][j + 1])
                    smooth -= abs(v - log2(b[i][j + 1]));
            }

    double mono = 0;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++) {
            mono -= abs(log2(b[i][j] + 1) - log2(b[i][j + 1] + 1));
            mono -= abs(log2(b[j][i] + 1) - log2(b[j + 1][i] + 1));
        }

    int corners[4] = {b[0][0], b[0][3], b[3][0], b[3][3]};
    int corner_bonus = (*max_element(corners, corners + 4) == maxT) ? maxT : 0;

    return
        2.7 * empty +
        smooth +
        mono +
        1.5 * corner_bonus +
        log2(maxT + 1);
}

// ===========================
// Expectimax
// ===========================
double expectimax(const Board& b, int depth, bool player,
                  chrono::steady_clock::time_point start,
                  double limit_sec) {

    if (depth == 0 ||
        chrono::duration<double>(chrono::steady_clock::now() - start).count() > limit_sec)
        return evaluate(b);

    if (player) {
        double best = -INF;
        for (int a = 0; a < 4; a++) {
            Board next;
            int r;
            if (!move_board(b, next, a, r)) continue;
            best = max(best, expectimax(next, depth - 1, false, start, limit_sec));
        }
        return (best == -INF) ? evaluate(b) : best;
    } else {
        vector<pair<int,int>> empty;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                if (b[i][j] == 0) empty.push_back({i,j});

        if (empty.empty()) return evaluate(b);

        double value = 0;
        double p = 1.0 / empty.size();
        for (auto [i,j] : empty) {
            for (auto [tile, prob] : vector<pair<int,double>>{{2,0.9},{4,0.1}}) {
                Board next = b;
                next[i][j] = tile;
                value += p * prob * expectimax(next, depth - 1, true, start, limit_sec);
            }
        }
        return value;
    }
}

// ===========================
// Choose action
// ===========================
int choose_action(const Board& b, double time_limit) {
    auto start = chrono::steady_clock::now();
    int best_action = 0;
    int depth = 2;

    while (true) {
        if (chrono::duration<double>(chrono::steady_clock::now() - start).count() > time_limit)
            break;

        double best = -INF;
        for (int a = 0; a < 4; a++) {
            Board next;
            int r;
            if (!move_board(b, next, a, r)) continue;
            double val = expectimax(next, depth, false, start, time_limit);
            if (val > best) {
                best = val;
                best_action = a;
            }
        }
        depth++;
    }
    return best_action;
}

// ===========================
// Spawn a new tile
// ===========================
void add_random_tile(Board& b, mt19937& rng) {
    vector<pair<int,int>> empty;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (b[i][j] == 0) empty.push_back({i,j});
    if (empty.empty()) return;

    uniform_int_distribution<int> dist_pos(0, empty.size()-1);
    uniform_real_distribution<double> dist_val(0.0,1.0);

    auto [i,j] = empty[dist_pos(rng)];
    b[i][j] = (dist_val(rng) < 0.9) ? 2 : 4;
}

// ===========================
// Print board
// ===========================
void print_board(const Board& b) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++)
            cout << b[i][j] << "\t";
        cout << "\n";
    }
}

// ===========================
// Check if any moves possible
// ===========================
bool can_move_any(const Board& b) {
    Board tmp;
    int r;
    for (int a = 0; a < 4; a++)
        if (move_board(b, tmp, a, r)) return true;
    return false;
}

// ===========================
// Main game loop
// ===========================
int main() {
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    Board board = {0};

    // start with 2 tiles
    add_random_tile(board, rng);
    add_random_tile(board, rng);

    int score = 0;
    int move_count = 0;

    while (can_move_any(board)) {
        cout << "Move #" << move_count << ":\n";
        print_board(board);
        cout << "Score: " << score << "\n\n";

        int action = choose_action(board, 1.0); // 30-second limit per move
        Board next;
        int reward;
        bool moved = move_board(board, next, action, reward);

        if (!moved) break; // no legal moves
        board = next;
        score += reward;
        add_random_tile(board, rng);
        move_count++;
    }

    cout << "Game Over\n";
    print_board(board);
    cout << "Final Score: " << score << "\n";
    cout << "Max Tile: " << max_tile(board) << "\n";

    return 0;
}