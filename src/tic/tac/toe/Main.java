package tic.tac.toe;


public class Main {
    
    public static void main(String[] args) {
        Game game = new TicTac();
        
        while(true) {
            game.playOneGame(2);
        }     
        
    }
}
