multiview_caption_prompt = """Here are some images of a single object, captured from different views. Please give a concise name and several properties  of the object, as well as a brief description of the object in the images. The description should be brief and at most 3-4 sentences long. 

You should integrate the information from all views. In some views, the object may be occluded by other objects. In such cases, you should still focus on the major object that appears at the center of most images.

The output format should be:
**Name**: <name of the object>
**Color**: <color of the object>
**Shape**: <shape of the object>
**Material**: <material of the object>
**Description**: <description of the object>"""

multi_render_caption_prompt = """Here are some images of a single object, captured from different views. Please give a concise name and several properties  of the object, as well as a brief description of the object in the images. The description should be brief and at most 3-4 sentences long. 

The images are rendered from a 3D model of the object. The images may be of low quality due to imperfect 3D reconstruction. You MUST always try your best to identify the object and give a best guess of the object name and properties. 

The output format should be:
**Name**: <name of the object>
**Color**: <color of the object>
**Shape**: <shape of the object>
**Material**: <material of the object>
**Description**: <description of the object>"""