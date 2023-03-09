sub_ais = ['first_settlement_ai', 'second_settlement_ai', 'initial_road_ai',
           'knight_ai', 'robber_ai', 'plenty_ai', 'monopoly_ai', 'development_ai', 'trade_ai', 'build_ai',
           'road_ai', 'settlement_ai', 'city_ai']
ai_choices = {sub_ai: ['basis', 'random'] for sub_ai in sub_ais}
ai_choices['road_ai'] += ['primitive_ml']
ai_choices['initial_road_ai'] += ['primitive_ml']
ai_choices['trade_ai'] += ['primitive_ml']