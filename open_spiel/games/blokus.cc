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

#include "open_spiel/games/blokus.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace blokus {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"blokus",
    /*long_name=*/"Blokus",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/4,
    /*min_num_players=*/4,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/ {}
    };

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BlokusGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

bool IsValidIndex(int i, int j) {
    if ((i < 0) || (i >= kNumRows) || (j < 0) || (j >= kNumCols)) {
        return false;
    }
    return true;
}

void Piece::UpdatePermutations(std::vector<Piece::shape>& perm, const Piece::shape& candidate) {
    if (std::find(perm.begin(), perm.end(), candidate) == perm.end()) {
        perm.push_back(candidate);
    }
}

std::vector<Piece::shape> Piece::GetPermutations() const {

    std::vector<Piece::shape> permutations;
    int max_size = MaxSize();

    Piece::shape rotated(piece_shape);
    Piece::shape flipped = FlipX(rotated, max_size);

    UpdatePermutations(permutations, rotated);
    UpdatePermutations(permutations, flipped);

    for (int i = 0; i < 3; ++i) {
        rotated = Rot90(rotated, max_size);
        flipped = FlipX(rotated, max_size);

        UpdatePermutations(permutations, rotated);
        UpdatePermutations(permutations, flipped);
    }

    return permutations;
}

int Piece::MaxSize() const {
    int max = 0;
    for (const auto& x : piece_shape) {
        max = std::max(max, x.first);
        max = std::max(max, x.second);
    }
    return max;
}

Piece::shape Piece::FlipX(const shape& sh, int max_size) {
    std::vector<Piece::point> mypoints;
    for (const auto& pt: sh) {
        mypoints.push_back({max_size - pt.first, pt.second});
    }
    Align(mypoints);
    return Piece::shape(mypoints.begin(), mypoints.end());
}

Piece::shape Piece::Rot90(const shape& sh, int max_size) {
    std::vector<Piece::point> mypoints;
    for (const auto& pt: sh) {
        mypoints.push_back({max_size - pt.second - 1, pt.first});
    }
    Align(mypoints);
    return Piece::shape(mypoints.begin(), mypoints.end());
}

void Piece::Align(std::vector<Piece::point>& sh) {
    int min_row = INT_MAX;
    int min_col = INT_MAX;

    for (const auto& pt: sh) {
        min_row = std::min(min_row, pt.first);
        min_col = std::min(min_col, pt.second);
    }

    for (auto &pt: sh) {
        pt.first -= min_row;
        pt.second -= min_col;
    }
}

// Move definitions

Move::Move(int piece_idx_, std::set<Move::point> positions, int x, int y, int move_idx_)
    : piece_idx(piece_idx_), move_idx(move_idx_) {

    for (const auto& loc : positions) {
        positions_.insert(std::make_pair(loc.first + x, loc.second + y));
    }

    GetNeighbors();
    GetCorners();
}

void Move::GetNeighbors() {
    Move::point candidate;
    
    for (const auto& pos : positions_) {
        for (const auto& dir : kNeighbors) {
            candidate = std::make_pair(pos.first + dir.first, pos.second + dir.second);
            if (IsValidIndex(candidate.first, candidate.second) &&
                std::find(positions_.begin(), positions_.end(), candidate) == positions_.end()) {
                neighbors_.insert(candidate);
            }
        }
    }
}

bool Move::CornerContainsNeighbor(Move::point& corner) const {
    Move::point candidate;

    for (const auto& dir : kNeighbors) {
        candidate = std::make_pair(corner.first + dir.first, corner.second + dir.second);
        if (positions_.find(candidate) != positions_.end()) {
            return true;
        }
    }
    return false;
}

void Move::GetCorners() {
    Move::point candidate;
    
    for (const auto& pos : positions_) {
        for (const auto& dir : kCorners) {
            candidate = std::make_pair(pos.first + dir.first, pos.second + dir.second);
            if (IsValidIndex(candidate.first, candidate.second) &&
                !CornerContainsNeighbor(candidate)) {
                corners_.insert(candidate);
            }
        }
    }
}

bool Move::SpaceTaken(const std::vector<std::vector<BlokusCell>>& board) const {
    for (const auto& pos: positions_) {
        if (board.at(pos.first).at(pos.second) != kEmpty) {
            return true;
        }
    }
    return false;
}

bool Move::ContainsNeighbor(const std::vector<std::vector<BlokusCell>>& board, BlokusCell cell) const {
    for (const auto& pos: neighbors_) {
        if (board.at(pos.first).at(pos.second) == cell) {
            return true;
        }
    }
    return false;
}

bool Move::ContainsCorner(const std::vector<std::vector<BlokusCell>>& board, BlokusCell cell) const {
    for (const auto& pos: corners_) {
        if (board.at(pos.first).at(pos.second) == cell) {
            return true;
        }
    }
    return false;
}

bool Move::IsValidMove(const std::vector<std::vector<BlokusCell>>& board, BlokusCell cell) const {
    return (!SpaceTaken(board) && !ContainsNeighbor(board, cell) && ContainsCorner(board, cell));
}

bool Move::IsValidFirstMove(const std::pair<int, int>& valid_move) const {
    return (positions_.find(valid_move) != positions_.end());
}

void Move::Apply(std::vector<std::vector<BlokusCell>>& board, BlokusCell cell) const {
    for (const auto& pos: positions_) {
        board.at(pos.first).at(pos.second) = cell;
    }

}

std::string Move::ToString() const {
    int count = 0;
    std::string str;
    absl::StrAppend(&str, "Positions: ");
    for (auto const& loc: positions_) {
        absl::StrAppend(&str, "(", loc.first, ", ", loc.second, ")");
        if (count != positions_.size() - 1) {
            absl::StrAppend(&str, ", ");
        }
        count += 1;
    }

    return str;
}

BlokusState::BlokusState(std::shared_ptr<const Game> game, const std::vector<Piece>& pieces, const std::vector<Move>& moves)
    : State(game), board_(kNumRows, std::vector<BlokusCell>(kNumCols, kEmpty)), pieces_(pieces), moves_(moves)  {

    // Initialize with all thue moves left
    std::fill(moves_left_.begin(), moves_left_.end(), kNumPieces);

    // First moves
    std::fill(first_move_.begin(), first_move_.end(), true);

    // If we are done
    std::fill(done_.begin(), done_.end(), false);
    
    // Initial score
    int initial_score = 0;
    for (const auto& p : pieces_) {
        initial_score += p.Size();
    }
    
    std::fill(scores_.begin(), scores_.end(), initial_score);

    // Opening moves
    init_moves_[0] = std::make_pair(kNumRows - 1, kNumCols - 1);
    init_moves_[1] = std::make_pair(kNumRows - 1, 0);
    init_moves_[2] = std::make_pair(0, 0);
    init_moves_[3] = std::make_pair(0, kNumCols - 1);
    
    // Initialize the valid pieces
    // TODO: Do this not horribly 
    for (auto &piece : valid_pieces_) {
        piece.fill(true);
    }
}


bool BlokusState::IsValidIndex(int i, int j) const {
    if ((i < 0) || (i >= kNumRows) || (j < 0) || (j >= kNumCols)) {
        return false;
    }
    return true;
}



bool BlokusState::FoundNeighbor(int player, const std::vector<std::pair<int, int>>& positions) const {
    std::pair<int, int> searchLoc;

    for (const auto& loc: positions) {
        for (const auto& offset: kNeighbors) {
            searchLoc = std::make_pair(loc.first + offset.first, 
                    loc.second + offset.second);

            if (IsValidIndex(searchLoc.first, searchLoc.second) &&
                std::find(positions.begin(), positions.end(), searchLoc) == positions.end() &&
                board_.at(searchLoc.first).at(searchLoc.second) == player) {
                return true;
            }
        }
    }

    return false;
}

bool BlokusState::ValidFirstMove(int player, const std::vector<std::pair<int, int>>& positions) const {
    if (std::find(positions.begin(), positions.end(), init_moves_[player]) != positions.end()) {
        return true;
    }
    return false;
}

// Might be more useful later on if need to switch indices, but keep for now
int BlokusState::PlayerToIndex(const BlokusCell cell) {
    return static_cast<int>(cell);
}

bool BlokusState::IsValidMove(BlokusCell player, const Move& move) const {
    int idx = PlayerToIndex(player);

    // Are we a first move
    if (first_move_[idx]) {
        if (move.IsValidFirstMove(init_moves_[idx])) {
            return true;
        }
        return false;
    }
    
    if (valid_pieces_.at(idx).at(move.piece_idx) && move.IsValidMove(board_, player)) {
        return true;
    }
    return false;
}

std::vector<Action> BlokusState::LegalActions() const {
    std::vector<Action> legal_moves;

    int player_idx = PlayerToIndex(current_player_);

    if (!IsTerminal()) {
        // Iterate over each cell in the board and check if the piece can fit
        
        for (const auto& move: moves_) {
            if (IsValidMove(current_player_, move)) {
                legal_moves.push_back(move.move_idx);
            }
        }
        
        // If no legal moves, do nothing
        if (legal_moves.size() == 0) {
            legal_moves.push_back(moves_.size());
        }
    }

    return legal_moves;
}

std::string BlokusState::ActionToString(Player player,
                                          Action action_id) const {

    SPIEL_CHECK_GE(action_id, 0);
    SPIEL_CHECK_LT(action_id, kNumDistinctActions);
    
    if (action_id == kNumDistinctActions - 1) {
        return "Null move";
    }
    
    int piece_idx = moves_[action_id].piece_idx;
    std::string piece_string = pieces_.at(piece_idx).ToString();
    std::string move_string = moves_.at(action_id).ToString();

    return piece_string + " at " + move_string;
}

std::string BlokusState::ToString() const {
    std::string str;
    std::string esc = "\033";
    std::string reset = esc + "[0m";
    std::string p1 = esc + "[1;33m" + "1" + reset;      // bright white
    std::string p2 = esc + "[1;34m" + "2" + reset;    // bright yellow
    std::string p3 = esc + "[1;35m" + "3" + reset;    // bright green
    std::string p4 = esc + "[1;36m" + "4" + reset;  // bright blue
    std::string empty = "0";

    for (const auto& row: board_) {
        for (const auto& col: row) {
            switch (col) {
                case kPlayer1 : absl::StrAppend(&str, p1, " "); break;
                case kPlayer2 : absl::StrAppend(&str, p2, " "); break;
                case kPlayer3 : absl::StrAppend(&str, p3, " "); break;
                case kPlayer4 : absl::StrAppend(&str, p4, " "); break;
                case kEmpty: absl::StrAppend(&str, empty, " "); break;
            }
        }
        absl::StrAppend(&str, "\n");
    }
    return str;
}

std::vector<double> BlokusState::Returns() const {
    if (outcome_ == kPlayer1) return {1, -1, -1, -1};
    if (outcome_ == kPlayer2) return {-1, 1, -1, -1};
    if (outcome_ == kPlayer3) return {-1, -1, 1, -1};
    if (outcome_ == kPlayer4) return {-1, -1, -1, 1};
    return {0, 0, 0, 0};
}

std::string BlokusState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string BlokusState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

int BlokusState::CellToMarker(BlokusCell cell) {
    if (cell == kEmpty) {
        return 0;
    }
    return static_cast<int>(cell + 1);
}

void BlokusState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(
          values, {kNumRows, kNumCols}, true);

  for (int i = 0; i < kNumRows; ++i) {
      for (int j = 0; j < kNumCols; ++j) {
          view[{i, j}] = CellToMarker(board_.at(i).at(j));
      }
  }

}

void BlokusState::DoApplyAction(Action action) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, moves_.size() + 1);

    int idx = PlayerToIndex(current_player_);

    if (action < moves_.size()) {
        Move move = moves_.at(action);
        
        SPIEL_CHECK_TRUE(IsValidMove(current_player_, move));

        move.Apply(board_, current_player_);

        valid_pieces_.at(idx).at(move.piece_idx) = false;
        moves_left_[idx] -= 1; // TODO: Catch  < 0
        first_move_[idx] &= false;
        scores_[idx] -= pieces_.at(move.piece_idx).Size();
    }

    // If we have a "do nothing" action or are out of moves, we are done
    if (!done_[idx] && (moves_left_[idx] == 0 || action == moves_.size())) {
        done_[idx] = true;
        num_done_++;
        if (num_done_ == kNumPlayers) {
            UpdateOutcome();
        }
    }

    // Update our current player
    current_player_ = static_cast<BlokusCell> ((current_player_ + 1) % kNumPlayers);
}

void BlokusState::UpdateOutcome() {
    std::unordered_map<int, int> counts;
    int winner = 0;
    int minimum = INT_MAX;

    // TODO: std::array
    for (int i = 0; i < scores_.size(); ++i) {
        if (scores_[i] <= minimum) { 
            if (counts.count(scores_[i]) == 0) {
                counts[scores_[i]] = 0;
            }
            counts[scores_[i]] += 1;
            minimum = scores_[i];
            winner = i + 1;
        }
    }

    if (counts.count(minimum) != 1) {
        winner = 0;
    }
    
    if (winner == 1) outcome_ = kPlayer1;
    if (winner == 2) outcome_ = kPlayer2;
    if (winner == 3) outcome_ = kPlayer3;
    if (winner == 4) outcome_ = kPlayer4;
}

std::unique_ptr<State> BlokusState::Clone() const {
  return std::unique_ptr<State>(new BlokusState(*this));
}

void BlokusState::UndoAction(Player player, Action move) {
    // TODO
}

BlokusGame::BlokusGame(const GameParameters& params) 
    : Game(kGameType, params) {

    InitializePieces();
    GenerateValidMoves();
}

void BlokusGame::InitializePieces() {
    using loc = std::pair<int, int>;
    std::vector<std::vector<loc>> peice;

    pieces_.push_back(Piece({{0, 0}}, "i1"));
    pieces_.push_back(Piece({{0, 0}, {1, 0}}, "i2"));
    pieces_.push_back(Piece({{0, 0}, {1, 0}, {2, 0}}, "i3"));
    pieces_.push_back(Piece({{0, 0}, {1, 0}, {2, 0}, {3, 0}}, "i4"));
    pieces_.push_back(Piece({{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}}, "i5"));
    pieces_.push_back(Piece({{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 3}}, "L5"));
    pieces_.push_back(Piece({{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1}}, "Y"));
    pieces_.push_back(Piece({{0, 0}, {0, 1}, {0, 2}, {1, 2}, {1, 3}}, "N"));
    pieces_.push_back(Piece({{0, 0}, {1, 0}, {1, 1}}, "V3"));
    pieces_.push_back(Piece({{0, 0}, {0, 1}, {1, 1}, {2, 0}, {2, 1}}, "U"));
    pieces_.push_back(Piece({{0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}}, "V5"));
    pieces_.push_back(Piece({{0, 0}, {1, 0}, {1, 1}, {2, 1}, {2, 2}}, "Z5"));
    pieces_.push_back(Piece({{0, 1}, {1, 0}, {1, 1}, {1, 2}, {2, 1}}, "X"));
    pieces_.push_back(Piece({{0, 0}, {0, 1}, {0, 2}, {1, 1}, {2, 1}}, "T5"));
    pieces_.push_back(Piece({{0, 0}, {1, 0}, {1, 1}, {2, 1}, {2, 2}}, "W"));
    pieces_.push_back(Piece({{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}}, "P"));
    pieces_.push_back(Piece({{0, 1}, {0, 2}, {1, 0}, {1, 1}, {2, 1}}, "F"));
    pieces_.push_back(Piece({{0, 0}, {0, 1}, {1, 0}, {1, 1}}, "O4"));
    pieces_.push_back(Piece({{0, 0}, {0, 1}, {0, 2}, {1, 2}}, "L4"));
    pieces_.push_back(Piece({{0, 0}, {0, 1}, {0, 2}, {1, 1}}, "T4"));
    pieces_.push_back(Piece({{0, 0}, {0, 1}, {1, 1}, {1, 2}}, "Z4"));

    SPIEL_CHECK_EQ(pieces_.size(), kNumPieces);
}

void BlokusGame::GenerateValidMoves() {
    int piece_idx = 0;
    int move_idx = 0;

    for (Piece &piece : pieces_) {
        for (auto permutation: piece.GetPermutations()) {
            for (int i = 0; i < kNumRows; ++i) {
                for (int j = 0; j < kNumCols; ++j) {
                    if (IsValidPermutation(i, j, permutation)) {
                        moves_.push_back(Move(piece_idx, permutation, i, j, move_idx));
                        move_idx++;
                    }
                }
            }
        }
        piece_idx++;
    }
}

bool BlokusGame::IsValidPermutation(int i, int j, std::set<std::pair<int, int>> rotation) const {
    for (const auto& coordinate: rotation) {
        if ((coordinate.first + i < 0) || (coordinate.first + i >= kNumRows) || 
            (coordinate.second + j < 0) || (coordinate.second + j >= kNumCols)) {
            return false;
        }
    }

    return true;
}


}  // namespace blokus
}  // namespace open_spiel
