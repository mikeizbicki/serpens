from Mundus.Object import *

def generate_knowledge_base(ram, ram2, **kwargs):
    kb = KnowledgeBase(observation_format={
        'objects_discrete': ['id'],
        'objects_discrete_id': ['id'],
        'objects_string_chunk': ['chunk_id'],
        'objects_continuous': ['relx', 'rely'],
        })

    if ram.mouse is not None:
        item = {}
        item['x'] = ram.mouse['x']
        item['y'] = ram.mouse['y']
        item['id'] = 0
        item['chunk_id'] = 'interface'
        kb['mouse'] = item

    # NOTE:
    # The NES uses the OAM region of memory $0200-$02FF for storing sprites;
    # this code is a generic object detector that extracts sprite information from this region of RAM
    #
    # See the following links for details on NES rendering/hardware:
    # <https://austinmorlan.com/posts/nes_rendering_overview/>
    # <https://www.copetti.org/writings/consoles/nes/>
    for i in range(64):
        # The following link describes the details of the information being extracted here
        # <https://www.nesdev.org/wiki/PPU_OAM>
        base_addr = 0x0200 + 4*i
        item = {}
        item['chunk_id'] = 'generic'
        item['id'] = ram[base_addr + 1]
        item['x'] = ram[base_addr + 3]
        item['y'] = ram[base_addr + 0]

        byte2 = ram[base_addr + 2]
        item['pal4'] = int(byte2 & 0x3 == 0)
        item['pal5'] = int(byte2 & 0x3 == 1)
        item['pal6'] = int(byte2 & 0x3 == 2)
        item['pal7'] = int(byte2 & 0x3 == 3)
        item['priority']= int(byte2 & 32  >  0)
        item['flip_hori'] = int(byte2 & 64  >  0)
        item['flip_vert'] = int(byte2 & 128 >  0)

        # only add the sprite if it is on screen
        if item['y'] < 240 and item['x'] < 240:
            kb[f'sprite_{i:02}'] = item

    # center and normalize all items
    kb.columns.add('relx')
    kb.columns.add('rely')
    for item, val in kb.items.items():
        kb.items[item]['relx'] = kb.items[item]['x'] - 120
        kb.items[item]['rely'] = kb.items[item]['y'] - 120

        # normalize
        kb.items[item]['relx'] /= 120
        kb.items[item]['rely'] /= 120

    # add events
    #kb.events = get_events(ram, ram2)

    return kb
