package tic.tac.toe;

import deepqnetwork.NeuralQLearner;
import java.util.ArrayList;
import org.jblas.DoubleMatrix;

public class TicTac extends Game {
    private final int BOARD_DIMENSION = 3;
    private final int WIN_REQUIREMENT = 3;
    int action1;
    int action2;
    int draws;
    int xWins;
    int oWins;
    private int movesDone;
    private DoubleMatrix state1;
    private DoubleMatrix state2;
    private Combo winnerCombo;
    private Block[] board;
    private ArrayList<Combo> combos;
    private NeuralQLearner player1;
    private NeuralQLearner player2;
    
    public TicTac() {
        player1 = new NeuralQLearner(9,9,0.95);
        player2 = new NeuralQLearner(9,9,0.95);
    }

    @Override
    void initializeGame() {
        winnerCombo = null;
        movesDone = 0;
        board = new Block[BOARD_DIMENSION * BOARD_DIMENSION];
        for (int i = 0; i < board.length; i++) {
            board[i] = new Block();
        }
        createCombos();
    }

    @Override
    void makePlay(int player) {
        
        movesDone++;
        if (player == 0) {
            state1 = getState();
            do {
                action1 = player1.getAction(state1);
                if (board[action1].value != 0)
                    player1.update(state1, action1, -1, state1);
            } while (board[action1].value != 0);
            drawX(action1);
            if (movesDone >= 3)
                player2.update(state2, action2, getPlayer2Reward(), getState());
        } else {
            state2 = getState();
            do {
                action2 = player2.getAction(state2);
                if (board[action2].value != 0)
                    player2.update(state2, action2, -1, state2);
            } while (board[action2].value != 0);
            drawO(action2);
            if (movesDone >= 2)
                player1.update(state1, action1, getPlayer1Reward(), getState());
        }
    }

    @Override
    boolean endOfGame() {
        for (Combo combo : combos) {
            if(combo.isComplete()) {
                winnerCombo = combo;
                return true;
            }
        }
        if (movesDone == 9) {
            return true;
        }
        return false;
    }

    @Override
    void printWinner() {
        if(winnerCombo == null){
            player1.update(state1, action1, 0, getState());
            player2.update(state2, action2, 0, getState());
            draws++;
        } else if (winnerCombo.tiles[0].value == 1) {
            player1.update(state1, action1, 1, getState());
            xWins++;
        } else {
            player2.update(state2, action2, 1, getState());
            oWins++;
        }
        System.out.println("X wins: " + xWins);
        System.out.println("O wins: " + oWins);
        System.out.println("Draws: " + draws);
        System.out.println();
    }
    
    private double getPlayer1Reward() {
        for (Combo combo : combos) {
            if(combo.isComplete() && combo.tiles[0].value == -1) {
                return -1;
            }
        }
        return 0;
    }
    
    private double getPlayer2Reward() {
        for (Combo combo : combos) {
            if(combo.isComplete() && combo.tiles[0].value == 1) {
                return -1;
            }
        }
        return 0;
    }
    
    private DoubleMatrix getState() {
        DoubleMatrix result = new DoubleMatrix(board.length, 1);
        for (int i = 0; i < board.length; i++) {
            result.put(i, board[i].value);
        }
        return result;
    }
    
    
    //Winning combos are made here
    private void createCombos() {
        combos = new ArrayList();
        // horizontal
        for (int y = 0; y < board.length; y += BOARD_DIMENSION) {
            combos.add(new Combo(board[y], board[y+1], board[y+2]));
        }

        // vertical
        for (int x = 0; x < WIN_REQUIREMENT; x++) {
            combos.add(new Combo(board[x], board[x+BOARD_DIMENSION], board[x+BOARD_DIMENSION * 2]));
        }

        // diagonals
        combos.add(new Combo(board[0], board[4], board[8]));
        combos.add(new Combo(board[2], board[4], board[6]));
    }
    
    
    
    private void drawX(int place) {
        board[place].value = 1;
    }
    
    private void drawO(int place) {
        board[place].value = -1;
    }
    
}
