import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage

if __name__ == '__main__':
    
    num_cores = util.get_num_CPU()

    # Q1.1.2
    """
    path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255
    filter_responses = visual_words.extract_filter_responses(image)
    util.display_filter_responses(filter_responses)
    """
    #===============================================================#

    # Q1.2
    #visual_words.compute_dictionary(num_workers=num_cores)
    
    #===============================================================#

    # Q1.3
    
    dictionary = np.load('dictionary.npy')

    # Image 1
    path_img1 = "../data/aquarium/sun_aairflxfskjrkepm.jpg"
    image1 = skimage.io.imread(path_img1)
    image1 = image1.astype('float')/255
    wordmap1 = visual_words.get_visual_words(image1, dictionary)

    # Image 2
    path_img2 = "../data/highway/sun_adlsedktxzdgqbdy.jpg"
    image2 = skimage.io.imread(path_img2)
    image2 = image2.astype('float')/255
    wordmap2 = visual_words.get_visual_words(image2, dictionary)

    # Image 3
    path_img3 = "../data/windmill/sun_ajbmlzwcgcjkjgbd.jpg"
    image3 = skimage.io.imread(path_img3)
    image3 = image3.astype('float')/255
    wordmap3 = visual_words.get_visual_words(image3, dictionary)

    fig = plt.figure(1)

    plt.subplot(231)
    plt.imshow(image1)
    plt.axis('off')
    plt.axis('equal')

    plt.subplot(234)
    plt.imshow(wordmap1)
    plt.axis('off')
    plt.axis('equal')

    plt.subplot(232)
    plt.imshow(image2)
    plt.axis('off')
    plt.axis('equal')

    plt.subplot(235)
    plt.imshow(wordmap2)
    plt.axis('off')
    plt.axis('equal')

    plt.subplot(233)
    plt.imshow(image3)
    plt.axis('off')
    plt.axis('equal')

    plt.subplot(236)
    plt.imshow(wordmap3)
    plt.axis('off')
    plt.axis('equal')

    plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.1, hspace = 0.1)
    visual_recog.get_feature_from_wordmap(wordmap1, 200)
    plt.show()
    
    #===============================================================#
    
    # Q2.1

    #===============================================================#
    
    
    

    #util.save_wordmap(wordmap, 'file.jpg')
    #visual_recog.build_recognition_system(num_workers=num_cores)

    #conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    #print(conf)
    #print(np.diag(conf).sum()/conf.sum())

    #vgg16 = torchvision.models.vgg16(pretrained=True).double()
    #vgg16.eval()
    #deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
    #conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
    #print(conf)
    #print(np.diag(conf).sum()/conf.sum())

