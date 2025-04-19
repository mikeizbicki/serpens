
# see <https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map>

def _ramstate_player_x(ram):
    return int(ram[0x03AD]) + 8

def _ramstate_player_y(ram):
    return int(ram[0x00CE])

def _ramstate_player_pos(ram):
    return int(ram[0x071D])

def _ramstate_powerup_state(ram):
    '''
    0 - Small
    1 - Big
    >2 - fiery
    '''
    return int(ram[0x0756])

def _ramstate_coins(ram):
    return int(ram[0x075E])

def _ramstate_lives(ram):
    return int(ram[0x075A])

def _ramstate_time(ram):
    return bcd_to_int(ram[0x07F8:0x07FA])

def _ramstate_score(ram):
    return bcd_to_int(ram[0x07DD:0x07E2])

def bcd_to_int(bcd_bytes):
    return sum(((bcd_bytes[i] >> 4) * 10 + (bcd_bytes[i] & 0x0F)) * 10**(len(bcd_bytes)-1-i) for i in range(len(bcd_bytes)))

