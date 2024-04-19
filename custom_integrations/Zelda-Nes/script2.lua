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


function all_enemies_killed2 ()
    if data.enemy_1_y - prev_enemy_1_y == 0 then
        static_frame = static_frames + 1
        return true
    else
        return false
    end
end

