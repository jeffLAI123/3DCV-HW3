import open3d as o3d
import numpy as np
import sys, os, argparse, glob
import multiprocessing as mp

import cv2

import time







class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

    def create_camera(self, R, t):
        camera = o3d.geometry.LineSet()
        camera.points = o3d.utility.Vector3dVector(
            [[0, 0, 0], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]])
        camera.lines = o3d.utility.Vector2iVector(
            [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
        camera.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (8, 1)))
        camera.rotate(R)
        camera.translate(t)
        return camera

    

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width = 1000, height=1000)
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    camera = self.create_camera(R, t)
                    vis.add_geometry(camera)
                    #TODO:
                    # insert new camera pose here using vis.add_geometry()
            except Exception as e:
                pass
                #print(e)

            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def process_frames(self, queue):
        start = time.time()



        # initialize
        bf = cv2.BFMatcher()
        #orb = cv2.ORB_create()
        cur_pose = np.eye(4, dtype=np.float64)
        sift = cv2.SIFT_create()


        pre_img = cv2.imread(self.frame_paths[0])
        #Frame 0 
        pre_frame_kp, pre_frame_des = sift.detectAndCompute(cv2.imread(self.frame_paths[0]), None)
        
        for idx,frame_path in enumerate(self.frame_paths[1:]):
            print("Frame {}".format(idx+1))
            #print(frame_path)
            
            img = cv2.imread(frame_path)
            #TODO: compute camera pose here
            
            cur_frame_kp, cur_frame_des = sift.detectAndCompute(img, None)
            matches = bf.knnMatch(cur_frame_des, pre_frame_des, k=2) 
            good_matches = []
            for m, n in matches:
       
                if m.distance < 0.75 * n.distance:
        
                    good_matches.append(m)
            good_matches = sorted(good_matches, key=lambda x: x.distance)


            #step2
            pre_frame_points = np.array([pre_frame_kp[m.trainIdx].pt for m in good_matches])
            cur_frame_points = np.array([cur_frame_kp[m.queryIdx].pt for m in good_matches])
            #step3

            E, mask = cv2.findEssentialMat(pre_frame_points, cur_frame_points, self.K,\
                                                            cv2.RANSAC, 0.999, 1.0)
            #print("E",E)
            
            
            #step4
            retval, R, t, mask, triangulated = cv2.recoverPose(E, pre_frame_points, cur_frame_points, self.K, distanceThresh=1000, mask = mask)
            triangulated = triangulated[:3,:] / triangulated[3,:].reshape(1,-1) # 3*N
            #print(triangulated[:,0])
            scale_factor = 1
            if idx != 0:
                scale_factor = self.get_scale_factor(pre_chosen_points, pre_frame_points, pre_triangulated, triangulated)
                if scale_factor > 2:
                    scale_factor = 2
                t = scale_factor * t
            else:
                scale_factor = 1
            
            #step5 
            #print("scale_factor",scale_factor)
            frame_pose = np.concatenate([R, t], -1)
            frame_pose = np.concatenate([frame_pose, np.zeros((1, 4))], 0)
            frame_pose[-1, -1] = 1.0

            cur_pose = cur_pose @ frame_pose

            R = cur_pose[:3, :3]
            t = cur_pose[:3, 3]
            



            #print(R,t)
            pre_chosen_points = cur_frame_points
            pre_frame_des, pre_frame_kp = cur_frame_des, cur_frame_kp
            pre_triangulated = triangulated
            
            #Send R,t to visualization process
            queue.put((R, t))
            img_show = cv2.drawKeypoints(img, cur_frame_kp, None, color=(0, 255, 0))
            cv2.imshow('frame', img_show)
            if cv2.waitKey(30) == 27: break
            #if idx == 1:break


        # timer
        print("The time used to execute this is given below")

        end = time.time()

        print(end - start)
            

    def get_scale_factor(self, pre_frame_points, cur_frame_points, pre_triangulated, triangulated):
        pre_triangulated = pre_triangulated.T
        triangulated = triangulated.T
        scale_factor = []
        same_idx = []
        for i in range(pre_frame_points.shape[0]):
            for j in range(cur_frame_points.shape[0]):
                if np.all(pre_frame_points[i] == cur_frame_points[j]):
                    same_idx.append([pre_triangulated[i] ,triangulated[j]])
        if len(same_idx) <= 1:
            return 1

        for i in range(2):
            idx = np.random.randint(len(same_idx), size=2)
            pre_dis = np.linalg.norm(same_idx[idx[0]][0] - same_idx[idx[1]][0])
            cur_dis = np.linalg.norm(same_idx[idx[0]][1] - same_idx[idx[1]][1])
            scale_factor.append(cur_dis / (pre_dis + 0.0001))
        return np.median(scale_factor)








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()


