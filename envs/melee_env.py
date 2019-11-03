import gym
import melee


class MeleeEnv(gym.Env):
    def __init__(self,
                 log=False,
                 self_play=True,
                 render=False,
                 iso_path='../smash.iso'):
        self.self_play = self_play
        self.logger = None

        if log:
            self.logger = melee.logger.Logger()

        opponent = melee.enums.ControllerType.GCN_ADAPTER

        if self_play:
            opponent = melee.enums.ControllerType.STANDARD

        self.dolphin = melee.dolphin.Dolphin(
                ai_port=1,
                opponent_port=2,
                opponent_type=opponent,
                logger=self.logger)
        self.gamestate = melee.gamestate.GameState(self.dolphin)
        self.player1 = melee.controller.Controller(port=1, dolphin=self.dolphin)
        self.p1_turn = True

        if self_play:
            self.player2 = melee.controller.Controller(port=2, dolphin=self.dolphin)

        self.dolphin.run(render=render, iso_path=iso_path)
        self.player1.connect()

        if self_play:
            self.player2.connect()

        self.reset()
    
    def step(self, action):
        raise NotImplementedError

    def reset(self):
        self.gamestate.step()

        while self.gamestate.menu_state not in [
                melee.enums.Menu.IN_GAME, melee.enums.Menu.SUDDEN_DEATH]:
            if self.gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT:
                melee.menuhelper.choosecharacter(
                        character=melee.enums.Character.FOX,
                        gamestate=self.gamestate,
                        port=1,
                        opponent_port=2,
                        controller=self.player1,
                        swag=(not self.self_play),
                        start=(not self.self_play))

                if self.self_play:
                    melee.menuhelper.choosecharacter(
                            character=melee.enums.Character.FOX,
                            gamestate=self.gamestate,
                            port=2,
                            opponent_port=1,
                            controller=self.player2,
                            start=True)
            elif self.gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
                melee.menuhelper.choosestage(
                        stage=melee.enums.Stage.FINAL_DESTINATION,
                        gamestate=self.gamestate,
                        controller=self.player1)
            elif self.gamestate.menu_state == melee.enums.Menu.POSTGAME_SCORES:
                melee.menuhelper.skippostgame(controller=self.player1)

                if self.self_play:
                    melee.menuhelper.skippostgame(controller=self.player2)

            self.player1.flush()

            if self.self_play:
                self.player2.flush()

            self.gamestate.step()

    def close(self):
        self.dolphin.terminate()

        if self.logger:
            self.logger.writelog()
