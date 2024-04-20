static_frames = 0

prev_enemy_6_direction = 0
prev_enemy_5_direction = 0
prev_enemy_4_direction = 0
prev_enemy_3_direction = 0
prev_enemy_2_direction = 0
prev_enemy_1_direction = 0

prev_enemy_6_x = 0
prev_enemy_5_x = 0
prev_enemy_4_x = 0
prev_enemy_3_x = 0
prev_enemy_2_x = 0
prev_enemy_1_x = 0

prev_enemy_6_y = 0
prev_enemy_5_y = 0
prev_enemy_4_y = 0
prev_enemy_3_y = 0
prev_enemy_2_y = 0
prev_enemy_1_y = 0

function all_enemies_killed ()
  if data.enemy_1_y - prev_enemy_1_y == 0 and
     data.enemy_2_y - prev_enemy_2_y == 0 and
     data.enemy_3_y - prev_enemy_3_y == 0 and
     data.enemy_4_y - prev_enemy_4_y == 0 and
     data.enemy_5_y - prev_enemy_5_y == 0 and
     data.enemy_6_y - prev_enemy_6_y == 0 and
     data.enemy_1_x - prev_enemy_1_x == 0 and
     data.enemy_2_x - prev_enemy_2_x == 0 and
     data.enemy_3_x - prev_enemy_3_x == 0 and
     data.enemy_4_x - prev_enemy_4_x == 0 and
     data.enemy_5_x - prev_enemy_5_x == 0 and
     data.enemy_6_x - prev_enemy_6_x == 0 then
    static_frames = static_frames + 1
  else
    static_frames = 0
  end
  prev_enemy_6_y = data.enemy_6_y
  prev_enemy_5_y = data.enemy_5_y
  prev_enemy_4_y = data.enemy_4_y
  prev_enemy_3_y = data.enemy_3_y
  prev_enemy_2_y = data.enemy_2_y
  prev_enemy_1_y = data.enemy_1_y
  prev_enemy_6_x = data.enemy_6_x
  prev_enemy_5_x = data.enemy_5_x
  prev_enemy_4_x = data.enemy_4_x
  prev_enemy_3_x = data.enemy_3_x
  prev_enemy_2_x = data.enemy_2_x
  prev_enemy_1_x = data.enemy_1_x
  return static_frames > 240
end

function link_hearts()
    if data.link_heart_partial == 0 then
        return 0
    end
    partial = 0.5
    if data.link_heart_partial > 127 then
        partial = 0
    end
    full = data.link_heart_full - math.floor(data.link_heart_full / 16) * 16
    return full - partial + 1
end

prev_hearts = -1
function diff_link_hearts()
    local new_hearts = link_hearts()
    local retvalue = 0
    if prev_hearts >= 0 then
        retvalue = new_hearts - prev_hearts
    end
    prev_hearts = new_hearts
    return retvalue
end

prev_killed_enemies = -1
function diff_killed_enemies()
    local new_killed_enemies = data.killed_enemies
    local retvalue = 0
    if data.killed_enemies == 0 then
        retvalue = 0
    elseif prev_killed_enemies >= 0 then
        retvalue = data.killed_enemies - prev_killed_enemies
    end
    prev_killed_enemies = new_killed_enemies
    return retvalue
end

prev_items_value = -1
function diff_items_value()
    local new_items_value = data.link_items_bombs + data.link_items_keys + data.link_items_rupees
    local retvalue = 0
    if prev_items_value >= 0 then
        retvalue = new_items_value - prev_items_value
    end
    prev_items_value = new_items_value
    return retvalue
end

prev_link_direction = -1
prev_link_sword = -1
function spam_penalty()
    local new_link_direction = data.link_direction
    local new_link_sword = math.floor(data.player1_buttons / 126)
    local retvalue = 0
    if prev_link_sword ~= 1 and new_link_sword == 1 then
        retvalue = retvalue + 1
    end
    if prev_link_direction >= 0 and prev_link_direction ~= new_link_direction then
        retvalue = retvalue + 1
    end
    prev_link_direction = new_link_direction
    prev_link_sword = new_link_sword
    return retvalue
end


function scenario_screenbattle_done()
    return link_hearts() == 0 or data.screen_mode ~= 5 or all_enemies_killed()
end

function scenario_screenbattle_reward()
    screen_change_penalty = 0
    if data.screen_mode ~= 5 then
        screen_change_penalty = -10
    end
    return diff_link_hearts() + diff_killed_enemies() + 0.1 * diff_items_value() + screen_change_penalty -- - 0.001 * spam_penalty() 
    -- return diff_killed_enemies()
end
