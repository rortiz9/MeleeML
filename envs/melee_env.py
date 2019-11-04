import gym
import melee


class MeleeEnv(gym.Env):
    def __init__(self,
                 log=False,
                 render=False,
                 self_play=False,
                 iso_path='../smash.iso'):
        self.logger = None
        self.self_play = self_play

        if log:
            self.logger = melee.logger.Logger()

        opponent = melee.enums.ControllerType.GCN_ADAPTER

        if self.self_play:
            opponent = melee.enums.ControllerType.STANDARD

        self.dolphin = melee.dolphin.Dolphin(
                ai_port=1,
                opponent_port=2,
                opponent_type=opponent,
                logger=self.logger)
        self.gamestate = melee.gamestate.GameState(self.dolphin)
        self.observation_space = gym.spaces.Discrete(len(self.gamestate.tolist()))
        self.action_space = gym.spaces.Discrete(len(melee.enums.Button))
        self.player1 = melee.controller.Controller(port=1, dolphin=self.dolphin)
        self.p1_turn = True

        if self.self_play:
            self.player2 = melee.controller.Controller(port=2, dolphin=self.dolphin)

        self.dolphin.run(render=render, iso_path=iso_path)
        self.player1.connect()

        if self_play:
            self.player2.connect()
    
    def step(self, action):
        self.gamestate.step()
        controller = self.player1
        
        if not self.p1_turn:
            controller = self.player2

        controller.tilt_analog(
                melee.enums.Button.BUTTON_MAIN, action[0], action[1])
        controller.tilt_analog(
                melee.enums.Button.BUTTON_C, action[2], action[3])

        if action[6]:
            controller.press_button(melee.enums.Button.BUTTON_A)
        if action[7]:
            controller.press_button(melee.enums.Button.BUTTON_B)
        if action[8]:
            controller.press_button(melee.enums.Button.BUTTON_X)
        if action[9]:
            controller.press_button(melee.enums.Button.BUTTON_Y)
        if action[10]:
            controller.press_button(melee.enums.Button.BUTTON_Z)
        if action[11]:
            controller.press_shoulder(melee.enums.Button.BUTTON_L, action[4])
        if action[12]:
            controller.press_shoulder(melee.enums.Button.BUTTON_R, action[5])
        if action[13]:
            controller.press_button(melee.enums.Button.BUTTON_START)
        if action[14]:
            controller.press_button(melee.enums.Button.BUTTON_D_UP)
        if action[15]:
            controller.press_button(melee.enums.Button.BUTTON_D_DOWN)
        if action[16]:
            controller.press_button(melee.enums.Button.BUTTON_D_LEFT)
        if action[17]:
            controller.press_button(melee.enums.Button.BUTTON_D_RIGHT)

        controller.flush()

        if self.logger:
            self.logger.logframe(self.gamestate)
            self.logger.writeframe()

        if self.self_play:
            self.p1_turn = not self.p1_turn

        self.gamestate.step()
        return self.gamestate.tolist()

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

        return self.gamestate.tolist()

    def close(self):
        self.dolphin.terminate()

        if self.logger:
            self.logger.writelog()
