use core::time;
use std::{
    fmt::Display,
    time::{Duration, Instant},
};

use hashbrown::HashMap;
use itertools::Itertools;
use rand::seq::IndexedRandom;
use crate::engine::Engine;

use super::board::*;

/// Function that evaluates a final board (draw, or no remaning actions for the current player).
pub fn white_score(board: &Board) -> f32 {
    debug_assert!(
        board.is_draw() || board.actions().is_empty(),
        "The board is not final"
    );
    if(board.is_draw()) {
        return 0.5;
    }
    else if(board.turn == Color::White){ 
        return 0.;
    }
    else{
        return 1.;
    }
}

/// Performs a single rollout and returns the evaluation of the final state.
pub fn rollout(board: &Board) -> f32 {
        let mut choosen : Option<Action> = board.actions().choose(&mut rand::rng()).cloned();
        let mut new_board :  Board = board.clone();
        let mut available = true;
        let mut random = &mut rand::rng();
        while(available){
            match choosen{
                Some(act)=>{
                    new_board = new_board.apply(&act);
                    choosen = new_board.actions().choose(&mut rand::rng()).cloned();
                } 
                None => available = false
            }
                
        }
        return white_score(&new_board);
}

/// Alias type to repesent a count of selections.
pub type Count = u64;

/// Node of the MCTS graph
struct Node {
    /// Board of the node
    board: Board,
    /// Numer of times this node has been selected
    count: Count,
    /// *All* valid actions available on the board, together with the number of times they have been selected (potentially 0)
    /// and the last known evaluation of the result board.
    /// The actions define the outgoing edges (the target nodes can be computed by applying the action on the board)
    out_edges: Vec<OutEdge>,
    /// Evaluation given by the initial rollout on expansion
    initial_eval: f32,
    /// Q(s): complete evaluation of the node (to be updated after each playout)
    eval: f32,
}

impl Node {
    /// Creates the node with a single evaluation from a rollout
    pub fn init(board: Board, initial_eval: f32) -> Node {
        // create one outgoing edge per valid action
        let out_edges = board
            .actions()
            .into_iter()
            .map(|a| OutEdge::new(a))
            .collect_vec();
        Node {
            board,
            count: 1,
            out_edges,
            initial_eval,
            eval: initial_eval,
        }
    }
}

/// Edge of the MCTS graph.
///
/// An `OutEdge` is attached to a node (source) and target can be computed by applying the action to the source.
struct OutEdge {
    // action of the edge
    action: Action,
    // N(s,a): number of times this edge was selected
    visits: Count,
    // Q(s,a): Last known evaluation of the board resulting from the action
    eval: f32,
}
impl OutEdge {
    /// Initializes a new edge for this actions (with a count and eval at 0)
    pub fn new(action: Action) -> OutEdge {
        OutEdge {
            action,
            visits: 0,
            eval: 0.,
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "\n{}", self.board)?;
        write!(f, "Q: {}    (N: {})\n", self.eval, self.count)?;
        // display edges by decreasing number of samples
        for OutEdge {
            action,
            visits,
            eval,
        } in self.out_edges.iter().sorted_by_key(|e| u64::MAX - e.visits)
        {
            write!(f, "{visits:>8} {action}   [{eval}]\n")?;
        }
        Ok(())
    }
}

pub struct MctsEngine {
    /// Graph structure
    nodes: HashMap<Board, Node>,
    /// weight given to the exploration term in UCB1
    pub exploration_weight: f32,
}
impl MctsEngine {
    pub fn new(exploration_weight: f32) -> MctsEngine {
        MctsEngine {
            nodes: HashMap::new(),
            exploration_weight,
        }
    }
}

impl MctsEngine {
    /// Selects the best action according to UCB1, or `None` if no action is available.
    pub fn select_ucb1(&self, board: &Board) -> Option<Action> {
        //debug_assert!(self.nodes.contains_key(board));
        //let mut choosen : Option<Action>;
        let T:f32;
        match board.turn {
            Color::Black => T=1.,
            Color::White =>T=-1.,
        }
        let mut max_ucb1:f32 = 0.;
        let mut return_action :Option<Action>= None;
        let node = self.nodes.get(board).unwrap();
        let n : f32 = node.count as f32;
        let mut q_sa : f32;
        let mut n_sa : f32;
        let mut c :f32;
        let mut sqrt:f32;
        let mut ucb1:f32;
        // a nettoyer plus tard : retirer les declarations useless
        for arc in node.out_edges.iter(){
            q_sa = arc.eval;
            n_sa = arc.visits as f32;
            c = self.exploration_weight;
            sqrt = 2.*(f32::log(n,10.)/n_sa).sqrt();
            ucb1 = T * q_sa + self.exploration_weight * sqrt;
            if (ucb1>max_ucb1){
                max_ucb1 = ucb1;
                return_action = Some(arc.action.clone());
            }
        }
        return return_action;
        
    }

    /// Performs a playout for this board (s) and returns the (updated) evaluation of the board (Q(s))
    fn playout(&mut self, board: &Board) -> f32 {
        let mut eval = rollout(board);
        if !self.nodes.contains_key(board) {
            self.nodes.insert(board.clone(), Node::init(board.clone(), eval));
            return eval
            
        } else {
            
            match (self.select_ucb1(board)){
                Some (action) =>{
                    let board_played = &board.apply(&action);
                    eval = self.playout(&board.apply(&action));
                    return self.update_eval(board, &action, eval)
                }
                None =>{
                    println!("Error: No action available for this board: {board}");
                    return eval;
                }
            }
            
        } ;
    }

    /// Updates the evaluation (Q(s)) of the board (s), after selected the action (a) for a new playout
    /// which yieled an evaluation of `action_eval` (Q(s,a))
    fn update_eval(&mut self, board: &Board, action: &Action, action_eval: f32) -> f32 {
        debug_assert!(self.nodes.contains_key(board));
        let node = self.nodes.get_mut(board).unwrap();
        let mut arc_store: &mut OutEdge;
        node.count += 1;
        node.eval = node.initial_eval/(node.count as f32) ;
        for arc in node.out_edges.iter_mut(){
            if arc.action.eq(action){
                arc.visits +=1;
                arc.eval = action_eval;
            }
            node.eval += ((arc.visits as f32)/(node.count as f32))* arc.eval;
        }
        return node.eval;
    }
}

impl Engine for MctsEngine {
    fn select(&mut self, board: &Board, deadline: Instant) -> Option<Action> {
        let time_remaining: bool = Instant::now() < deadline;
        while(time_remaining){
            let time_remaining: bool = Instant::now() < deadline;
            self.playout(board);
        }
        return todo;
    }

    fn clear(&mut self) {
        self.nodes.clear();
    }
}

#[cfg(test)]
mod test {
    use crate::Color;

    use super::{Board, MctsEngine};

    #[test]
    fn test_mcts() {
        let board = Board::parse(
            "
              ABCDEFGH   White  (32 plies)
            1  b . b b
            2 . . . b
            3  . . . w
            4 . . . .
            5  . . . .
            6 . b w .
            7  . . w .
            8 w w w .",
            Color::White,
        );
        let mut mcts = MctsEngine::new(1.);

        println!("{board}");

        for i in 1..=4 {
            mcts.playout(&board);
            println!("After {i} playouts: \n{}", mcts.nodes[&board]);
        }
        println!("{board}");
    }
}
