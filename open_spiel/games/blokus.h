// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_BLOKUS_H_
#define OPEN_SPIEL_GAMES_BLOKUS_H_

#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Blokus
// https://en.wikipedia.org/wiki/Blokus
//
// Parameters: none

namespace open_spiel {
namespace blokus {

// Constants.
inline constexpr int kNumPlayers = 4;
inline constexpr int kNumRows = 20;
inline constexpr int kNumCols = 20;
inline constexpr int kBoardSize = kNumRows * kNumCols;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kCellStates = 1 + kNumPlayers;  // empty + the 4 players available
inline constexpr int kNumPieces = 21;
inline constexpr int kNumDistinctActions = 30434; // TODO: Double check this
inline constexpr std::array<std::pair<int, int>, 4> kNeighbors = {{{0, 1}, {0, -1}, {-1, 0}, {1, 0}}};
inline constexpr std::array<std::pair<int, int>, 4> kCorners = {{{-1, 1}, {-1, -1}, {1, -1}, {1, 1}}};

// State of a blokus cell
enum BlokusCell : uint8_t {
  kPlayer1,
  kPlayer2,
  kPlayer3,
  kPlayer4,
  kEmpty,
};

// What positions the piece holds, use top left position as anchor point
// Contains a vector of rotations
// TODO: Automate rotation code
struct Piece {
    using point = std::pair<int, int>;
    using shape = std::set<point>;
    
    const shape piece_shape;
    const std::string name;

    Piece(shape shape_, std::string name_) 
        : name(name_), piece_shape(shape_) {};
    
    std::string ToString() const { return name; };
    std::vector<shape> GetPermutations() const;
    int Size() const { return static_cast<int>(piece_shape.size()); };
    int MaxSize() const;
    static void UpdatePermutations(std::vector<shape>&, const shape&);
    static shape FlipX(const shape&, int);
    static shape Rot90(const shape&, int);
    static void Align(std::vector<point>&);
};

// A move consists of a piece and it's rotation
class Move {
    using point = std::pair<int, int>;

    public:
        Move(int piece_idx, std::set<point> permutations, int x_, int y_, int move_idx_);
        bool IsValidFirstMove(const point&) const;
        bool IsValidMove(const std::vector<std::vector<BlokusCell>>&, BlokusCell) const;
        void Apply(std::vector<std::vector<BlokusCell>>&, BlokusCell) const;
        std::string ToString() const;
        const int piece_idx;
        const int move_idx;
    
    private:
        std::set<point> positions_;
        std::set<point> neighbors_;
        std::set<point> corners_;

        void GetNeighbors();
        bool CornerContainsNeighbor(point&) const;
        void GetCorners();

        bool SpaceTaken(const std::vector<std::vector<BlokusCell>>&) const;
        // TODO: Merge definitions
        bool ContainsNeighbor(const std::vector<std::vector<BlokusCell>>&, BlokusCell cell) const;
        bool ContainsCorner(const std::vector<std::vector<BlokusCell>>&, BlokusCell cell) const;
};

// State of an in-play game.
class BlokusState : public State {
    public:
        BlokusState(std::shared_ptr<const Game> game);

        BlokusState(const BlokusState&) = default;
        BlokusState& operator=(const BlokusState&) = default;

        Player CurrentPlayer() const override {
            return IsTerminal() ? kTerminalPlayerId : static_cast<int>(current_player_);
        }
        std::string ActionToString(Player player, Action action_id) const override;
        std::string ToString() const override;
        bool IsTerminal() const override { return num_done_ == kNumPlayers; }
        std::vector<double> Returns() const override;
        std::string InformationStateString(Player player) const override;
        std::string ObservationString(Player player) const override;
        void ObservationTensor(Player player,
                             absl::Span<float> values) const override;
        std::unique_ptr<State> Clone() const override;
        void UndoAction(Player player, Action move) override;
        std::vector<Action> LegalActions() const override;

    protected:
        void DoApplyAction(Action move) override;
        void InitializePieces();
        void GenerateValidMoves();
        bool IsValidPermutation(int i, int j, std::set<std::pair<int, int>> rotation) const;
        bool IsValidMove(BlokusCell current_player, const Move& move) const;
        bool IsValidIndex(int i, int j) const;
        bool FoundNeighbor(int player, const std::vector<std::pair<int, int>>& positions) const;
        bool ValidFirstMove(int player, const std::vector<std::pair<int, int>>& positions) const;
        void UpdateOutcome();
        static int CellToMarker(BlokusCell);
        static int PlayerToIndex(const BlokusCell);
    
    private:
        BlokusCell current_player_ = kPlayer1;
        Player outcome_ = kEmpty;
        int num_done_ = 0;
        std::vector<std::vector<BlokusCell>> board_;
        std::vector<Piece> pieces_;

        std::vector<Move> moves_;
        std::unordered_map<std::string, int> string_to_action;

        std::array<std::array<bool, kNumPieces>, kNumPlayers> valid_pieces_;

        // TODO: make this an object?
        std::array<int, kNumPlayers> moves_left_;
        std::array<bool, kNumPlayers> first_move_;
        std::array<bool, kNumPlayers> done_;
        std::array<int, kNumPlayers> scores_;
        std::array<std::pair<int, int>, kNumPlayers> init_moves_;
};

// Game object.
class BlokusGame : public Game {
 public:
  explicit BlokusGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; };
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new BlokusState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kNumRows, kNumCols}; // 2d tensor
  }
  int BoardSize() const { return kBoardSize; };
  int MaxGameLength() const override { return kNumPieces * kNumPlayers; }
};

}  // namespace blokus
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BLOKUS_H_
