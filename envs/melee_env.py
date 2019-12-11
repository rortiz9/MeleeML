import gym
import melee
import numpy as np
from envs.dataset import preprocess_states

class MeleeEnv(gym.Env):
    def __init__(self,
                 action_set,
                 log=False,
                 render=False,
                 self_play=False,
                 iso_path='../smash.iso'):
        self.logger = None
        self.self_play = self_play
        self.first_time_in_menu = True
        self.action_set = action_set
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
        state = preprocess_states(state)
        return state

    def step(self, action):
        self.gamestate.step()
        action = self._one_hot_to_action(action, self.action_set)
        controller = self.player1

        if not self.p1_turn:
            controller = self.player2

        controller.tilt_analog(
                melee.enums.Button.BUTTON_MAIN, action[2], action[3])
        controller.tilt_analog(
                melee.enums.Button.BUTTON_C, action[4], action[5])

        button = list()

        if action[6]:
            button.append(melee.enums.Button.BUTTON_A)
        if action[7]:
            button.append(melee.enums.Button.BUTTON_B)
        if action[8]:
            button.append(melee.enums.Button.BUTTON_X)
        if action[9]:
            button.append(melee.enums.Button.BUTTON_Y)
        if action[10]:
            button.append(melee.enums.Button.BUTTON_Z)
        if action[11]:
            button.append(melee.enums.Button.BUTTON_D_UP)
        if action[12]:
            button.append(melee.enums.Button.BUTTON_D_DOWN)
        if action[13]:
            button.append(melee.enums.Button.BUTTON_D_LEFT)
        if action[14]:
            button.append(melee.enums.Button.BUTTON_D_RIGHT)

        if action[15]:
            controller.press_shoulder(melee.enums.Button.BUTTON_L, action[0])
            controller.press_shoulder(melee.enums.Button.BUTTON_R, action[1])
        else:
            controller.press_shoulder(melee.enums.Button.BUTTON_L, 0)
            controller.press_shoulder(melee.enums.Button.BUTTON_R, 0)

        for item in melee.enums.Button:
            if item in [
                    melee.enums.Button.BUTTON_MAIN,
                    melee.enums.Button.BUTTON_C,
                    melee.enums.Button.BUTTON_L,
                    melee.enums.Button.BUTTON_R]:
                continue

            if item in button:
                controller.press_button(item)
            else:
                controller.release_button(item)

        controller.flush()
        self.gamestate.step()
        state = self._strip_state(self.gamestate.tolist())

        done = False
        p1_score = self.p1_percent - self.gamestate.ai_state.percent
        p2_score = self.p2_percent - self.gamestate.opponent_state.percent

        if self.p1_stock > self.gamestate.ai_state.stock:
            p1_score = -1000 * (self.p1_stock - self.gamestate.ai_state.stock)
            self.p1_stock = self.gamestate.ai_state.stock
        if self.p2_stock > self.gamestate.opponent_state.stock:
            p2_score = -1000 * (self.p2_stock - self.gamestate.opponent_state.stock)
            self.p2_stock = self.gamestate.opponent_state.stock

        self.p1_percent = self.gamestate.ai_state.percent
        self.p2_percent = self.gamestate.opponent_state.percent

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
        self.p1_stock = 4
        self.p1_percent = 0.0
        self.p2_stock = 4
        self.p2_percent = 0.0
        self.gamestate.step()
        count = 0

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
                count += 1
                if not self.first_time_in_menu and count%10  == 2:
                    melee.menuhelper.skippostgame(controller=self.player1)

            elif self.gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
                melee.menuhelper.choosestage(
                        stage=melee.enums.Stage.FINAL_DESTINATION,
                        gamestate=self.gamestate,
                        controller=self.player1)
                self.first_time_in_menu = False

            elif self.gamestate.menu_state == melee.enums.Menu.POSTGAME_SCORES:
                melee.menuhelper.skippostgame(controller=self.player1)

                if self.self_play:
                    melee.menuhelper.skippostgame(controller=self.player2)

            self.player1.flush()

            if self.self_play:
                self.player2.flush()

            self.gamestate.step()

        return self._strip_state(self.gamestate.tolist())

    # reformats one hot action
    def _one_hot_to_action(one_hot_action, action_set):
        code_to_shield = {0: 0, 1: 65/255., 2: 150/255.}
        code_to_analog = {0: 0, 1: 55/255., 2: 127/255., 3: 155/255., 4: 205/255.}
        code_to_c_stick = {0: (127/255., 127/255.), 1: (50/255., 127/255.), 2: (50/255., 50/255.),
                            3: (50, 205/255.),
                           4: (205/255., 127/255.), 5: (205/255., 50/255.), 6: (205/255., 205/255.),
                           7: (127/255.,254/255.),  8: (127/255., 1/255.)}
        intermediate_action = action_set[np.where(one_hot_action == 1)[0]]
        # see dataset.py for structure for intermediate action
        action = np.zeros((16))
        '''action[:2] = intermediate_action[:2]
        action[2] = intermediate_action[2]
        action[3] = intermediate_action[3]
        action[10] = code_to_shield[intermediate_action[4]]
        action[12] = code_to_analog[intermediate_action[5]]
        action[13] = code_to_analog[intermediate_action[6]]
        c_stick = code_to_c_stick[intermediate_action[7]]
        action[14] = c_stick[0]
        action[15] = c_stick[1]'''
        action[0] = code_to_shield[intermediate_action[4]]
        action[2] = code_to_analog[intermediate_action[5]]
        action[3] = code_to_analog[intermediate_action[6]]
        c_stick = code_to_c_stick[intermediate_action[7]]
        action[4] = c_stick[0]
        action[5] = c_stick[1]
        action[6] = intermediate_action[0]
        action[7] = intermediate_action[1]
        action[8] = intermediate_action[2]
        action[10] =  intermediate_action[3]
        action[15] =  1

        return action

    def close(self):
        self.dolphin.terminate()

        if self.logger:
            self.logger.writelog()
