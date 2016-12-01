package tic.tac.toe;


//Combos which lead to winning the game

public class Combo {
        Block[] tiles;
        public Combo(Block... tiles) {
            this.tiles = tiles;
        }
        public boolean isComplete() {
            if (tiles[0].value == 0)
                return false;
            
            return tiles[0].value == tiles[1].value
                    && tiles[0].value == tiles[2].value;
        }
    }
