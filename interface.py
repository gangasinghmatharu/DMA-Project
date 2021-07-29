# ---------------------------------------Interface part-------------------------------------------

def lab_to_rgb(L, ab):
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def result(inp):
    data = Image.fromarray(inp) # convert array to image
    data.save('/content/drive/My Drive/test.jpeg')  # save input image
    path = '/content/drive/My Drive/test.jpeg'
    input_img = CreateDataloader([path, path],'validation') # load input image
    data = next(iter(input_img))
    with torch.no_grad():
        model.setupInputValues(data)
        model.forward()
    fake_color = model.fake_color.detach()
    L = model.L
    fake_img = lab_to_rgb(L, fake_color)
    return fake_img[0]  # return fake image

# input image
input_image = gr.inputs.Image()
# output image
output_image = gr.outputs.Image()

# GUI
gr.Interface(fn=result, inputs=input_image, outputs=output_image, capture_session=True).launch(debug=True)

