from Mundus.Object import *

def generate_knowledge_base(ram, ram2, include_background=True, use_subtiles=False):
    kb = KnowledgeBase(keys={
        'objects_discrete': ['type', 'direction', 'state'],
        'objects_continuous': ['relx', 'rely', 'x', 'y', 'dx', 'dy', 'health'],
        })

    if ram.mouse is not None:
        item = {}
        item['x'] = ram.mouse['x']
        item['y'] = ram.mouse['y']
        item['dx'] = ram.mouse['x'] - ram2.mouse['x'] if ram2 is not None and ram2.mouse is not None else 0
        item['dy'] = ram.mouse['y'] - ram2.mouse['y'] if ram2 is not None and ram2.mouse is not None else 0
        item['type'] = -5
        item['state'] = 0
        item['direction'] = 0
        item['health'] = 0
        kb['mouse'] = item

    # FIXME:
    # The NES uses the OAM region of memory $0200-$02FF for storing sprites;
    # we could have a generic object detector just by using this region of ram
    # see:
    # <https://austinmorlan.com/posts/nes_rendering_overview/>
    # <https://www.nesdev.org/wiki/PPU_OAM>
    #
    # good overview of allthings NES:
    # <https://www.copetti.org/writings/consoles/nes/>
    first_sprite = None
    for i in range(64):
        base_addr = 0x0200 + 4*i
        item = {}
        item['state'] = ram[base_addr + 2]
        item['direction'] = 0
        item['type'] = ram[base_addr + 1]
        item['x'] = ram[base_addr + 3]
        item['y'] = ram[base_addr + 0]
        item['dx'] = ram[base_addr + 3] - ram2[base_addr + 3] if ram2 is not None else 0
        item['dy'] = ram[base_addr + 0] - ram2[base_addr + 0] if ram2 is not None else 0
        item['health'] = 0

        # only add the sprite if it is on screen
        if item['y'] < 240 and item['x'] < 240:
            kb[f'sprite_{i:02}'] = item

            if first_sprite is None:
                first_sprite = item


    # normalize and center all item positions
    kb.columns.add('relx')
    kb.columns.add('rely')
    for item, val in kb.items.items():
        # center on first_sprite
        kb.items[item]['relx'] = kb.items[item]['x'] - first_sprite['x']
        kb.items[item]['rely'] = kb.items[item]['y'] - first_sprite['y']

        # normalize
        kb.items[item]['relx'] /= 240
        kb.items[item]['rely'] /= 160
        # NOTE:
        # the NES x resolution is 256,
        # but all sprites are 16 pixels wide,
        # so there are 240 available for positioning;
        # the y resolution is 224,
        # but the top vertical bar uses 56 pixels,
        # and only 8 pixels are reserved for "link positioning"
        # because he does not go "partially off the edge" on the top screen

    # add events
    #kb.events = get_events(ram, ram2)

    return kb


