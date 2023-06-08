class_tree_cover = '#006400'
class_tree_cover = class_tree_cover.lstrip('#')
class_tree_cover = np.array(tuple(int(class_tree_cover[i:i+2], 16) for i in (0,2,4)))
print(class_tree_cover)

class_shrubland = '#8429F6'
class_land = class_land.lstrip('#')
class_land = np.array(tuple(int(class_land[i:i+2], 16) for i in (0,2,4)))
print(class_land)

class_road = '#6EC1E4'
class_road = class_road.lstrip('#')
class_road = np.array(tuple(int(class_road[i:i+2], 16) for i in (0,2,4)))
print(class_road)

class_vegetation = '#FEDD3A'
class_vegetation = class_vegetation.lstrip('#')
class_vegetation = np.array(tuple(int(class_vegetation[i:i+2], 16) for i in (0,2,4)))
print(class_vegetation)

class_water = '#E2A929'
class_water = class_water.lstrip('#')
class_water = np.array(tuple(int(class_water[i:i+2], 16) for i in (0,2,4)))
print(class_water)

class_unlabeled = '#9B9B9B'
class_unlabeled = class_unlabeled.lstrip('#')
class_unlabeled = np.array(tuple(int(class_unlabeled[i:i+2], 16) for i in (0,2,4)))
print(class_unlabeled)


10 #006400 Tree cover
20 #ffbb22 Shrubland
30 #ffff4c Grassland
40 #f096ff Cropland
50 #fa0000 Built-up
60 #b4b4b4 Bare / sparse vegetation
70 #f0f0f0 Snow and ice
80 #0064c8 Permanent water bodies
90 #0096a0 Herbaceous wetland
95 #00cf75 Mangroves
100 #fae6a0 Moss and lichen



labels_dict = {"classes": [
{"title": "Tree cover", "r": 0 , "g": 0 , "b": 0 }, 
{"title": "Shrubland", "r": 132, "g": 41, "b": 246 }, 
{"title": "Grassland", "r": 110, "g": 193, "b": 228 }, 
{"title": "Cropland", "r": 60, "g": 16, "b": 152 }, 
{"title": "Built-up", "r": 254, "g": 221, "b": 58 }, 
{"title": "Bare / sparse vegetation", "r": 155, "g": 155, "b": 155 }
{"title": "Snow and ice", "r": 155, "g": 155, "b": 155 }
{"title": "Permanent water bodies", "r": 155, "g": 155, "b": 155 }
{"title": "Herbaceous wetland", "r": 155, "g": 155, "b": 155 }
{"title": "Mangroves", "r": 155, "g": 155, "b": 155 }
{"title": "Moss and lichen", "r": 155, "g": 155, "b": 155 }
]}

labels_dict_df = pd.DataFrame(labels_dict['classes'])

hex_colors_list = ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000',
                   '#b4b4b4', '#f0f0f0', '#0064c8', '#0096a0', '#00cf75', '#fae6a0']

labels_dict_2 = {'label_10': '#006400', 
                 'label_20': '#ffbb22',
                 'label_30':  '#ffff4c',
                 'label_40': '#f096ff',
                 'label_50' : '#fa0000',
                 'label_60' : '#b4b4b4',
                 'label_70' : '#f0f0f0',
                 'label_80' : '#0064c8',
                 'label_90' : '#0096a0',
                 'label_95' : '#00cf75',
                 'label_100' : '#fae6a0'}

for i in range(len(hex_colors_list)):
    color = hex_colors_list[i].lstrip('#')
    r = int(color[0:2],16)
    g = int(color[2:4],16)
    b = int(color[4:6],16)
    labels_dict_df.at[i,'r'] = r
    labels_dict_df.at[i,'g'] = g
    labels_dict_df.at[i,'b'] = b
    
     