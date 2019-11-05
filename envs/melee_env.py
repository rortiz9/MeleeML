import gym
import melee
import numpy as np


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
        self.player1 = melee.controller.Controller(port=1, dolphin=self.dolphin)
        self.p1_turn = True

        if self.self_play:
            self.player2 = melee.controller.Controller(port=2, dolphin=self.dolphin)

        self.dolphin.run(render=render, iso_path=iso_path)
        self.player1.connect()

        if self_play:
            self.player2.connect()

    def _strip_state(self, state):
        state = [self.gamestate.stage.value] + state
        state = np.array(state)
        state = np.delete(state, [10, 11, 27, 28])
        return state
    
    def step(self, action):
        self.gamestate.step()

        p1_stock = self.gamestate.ai_state.stock
        p1_percent = self.gamestate.ai_state.percent
        p2_stock = self.gamestate.opponent_state.stock
        p2_percent = self.gamestate.opponent_state.percent

        controller = self.player1

        if not self.p1_turn:
            controller = self.player2

        controller.tilt_analog(
                melee.enums.Button.BUTTON_MAIN, action[2], action[3])
        controller.tilt_analog(
                melee.enums.Button.BUTTON_C, action[4], action[5])
        controller.press_shoulder(melee.enums.Button.BUTTON_L, action[0])
        controller.press_shoulder(melee.enums.Button.BUTTON_R, action[1])

        button = None

        if action[6]:
            button = melee.enums.Button.BUTTON_A
        elif action[7]:
            button = melee.enums.Button.BUTTON_B
        elif action[8]:
            button = melee.enums.Button.BUTTON_X
        elif action[9]:
            button = melee.enums.Button.BUTTON_Y
        elif action[10]:
            button = melee.enums.Button.BUTTON_Z
        elif action[11]:
            button = melee.enums.Button.BUTTON_D_UP
        elif action[12]:
            button = melee.enums.Button.BUTTON_D_DOWN
        elif action[13]:
            button = melee.enums.Button.BUTTON_D_LEFT
        elif action[14]:
            button = melee.enums.Button.BUTTON_D_RIGHT

        for item in melee.enums.Button:
            if item == melee.enums.Button.BUTTON_MAIN:
                continue
            if item == melee.enums.Button.BUTTON_C:
                continue

            if item == button:
                controller.press_button(item)
            else:
                controller.release_button(item)

        controller.flush()
        self.gamestate.step()
        state = self._strip_state(self.gamestate.tolist())

        done = False
        p1_score = (1000 * (self.gamestate.ai_state.stock - p1_stock) -
                   (self.gamestate.ai_state.percent - p1_percent))
        p2_score = (1000 * (self.gamestate.opponent_state.stock - p2_stock) -
                   (self.gamestate.opponent_state.percent - p2_percent))

        if (self.gamestate.ai_state.stock == 0
                or self.gamestate.opponent_state.stock == 0):
            done = True

        reward = p1_score - p2_score

        if not self.p1_turn:
            state = []
            state.append(self.gamestate.stage.value)
            state = state + self.gamestate.opponent_state.tolist()
            state = state + self.gamestate.ai_state.tolist()
            state = self._strip_state(state)
            reward = p2_score - p1_score

        if self.self_play:
            self.p1_turn = not self.p1_turn

        if self.logger:
            self.logger.logframe(self.gamestate)
            self.logger.writeframe()

        return state, reward, done

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
                        controller=self.player1)

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

        return self._strip_state(self.gamestate.tolist())

    def close(self):
        self.dolphin.terminate()

        if self.logger:
            self.logger.writelog()
