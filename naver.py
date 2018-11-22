import torch
import torch.nn as nn


class HgResBlock(nn.Module):
    ''' Hourglass residual block '''
    def __init__(self, inplanes, outplanes, stride=1):
        super().__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        midplanes = outplanes // 2
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, midplanes, 1, stride)  # bias=False
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(midplanes, midplanes, 3, stride, 1)
        self.bn3 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, 1, stride)  # bias=False
        self.relu = nn.ReLU(inplace=True)
        if inplanes != outplanes:
            self.conv_skip = nn.Conv2d(inplanes, outplanes, 1, 1)
 
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.inplanes != self.outplanes:
            residual = self.conv_skip(residual)
        out += residual
        return out

class Hourglass(nn.Module):
    def __init__(self, depth, nFeat, nModules, resBlock):
        super().__init__()
        self.depth = depth
        self.nFeat = nFeat
        self.nModules = nModules  # num residual modules per location
        self.resBlock = resBlock
 
        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
 
    def _make_hour_glass(self):
        hg = []
        for i in range(self.depth):
            res = [self._make_residual(self.nModules) for _ in range(3)]  # skip(upper branch); down_path, up_path(lower branch)
            if i == (self.depth - 1):
                res.append(self._make_residual(self.nModules))  # extra one for the middle
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)
 
    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])
 
    def forward(self, x):
        return self._hour_glass_forward(0, x)
 
    def _hour_glass_forward(self, depth_id, x):
        up1 = self.hg[depth_id][0](x)
        low1 = self.downsample(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == (self.depth - 1):
            low2 = self.hg[depth_id][3](low1)
        else:
            low2 = self._hour_glass_forward(depth_id + 1, low1)
        low3 = self.hg[depth_id][2](low2)
        up2 = self.upsample(low3)
        return up1 + up2
 

class Hourglass2(nn.Module):
    def __init__(self, depth, nFeat, nModules, resBlock):
        super().__init__()
        self.depth = depth
        self.nFeat = nFeat
        self.nModules = nModules  # num residual modules per location
        self.resBlock = resBlock
 
        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
 
    def _make_hour_glass(self):
        hg = []
        for i in range(self.depth):
            res = [self._make_residual(self.nModules) for _ in range(3)]  # skip(upper branch); down_path, up_path(lower branch)
            if i == (self.depth - 1):
                res.append(self._make_residual(self.nModules))  # extra one for the middle
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)
 
    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])
 
    def forward(self, x):
        return self._hour_glass_forward(0, x)
 
    def _hour_glass_forward(self, depth_id, x):
        up1 = self.hg[depth_id][0](x)
        low1 = self.downsample(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == (self.depth - 1):
            low2 = self.hg[depth_id][3](low1)
        else:
            low2 = self._hour_glass_forward(depth_id + 1, low1)
        low3 = self.hg[depth_id][2](low2)
        up2 = self.upsample(low3)
        return up1 + up2, low1, low2, low3
 

class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, nStacks, nModules, nFeat, nClasses, resBlock=HgResBlock, inplanes=3):
        super().__init__()
        self.nStacks = nStacks
        self.nModules = nModules
        self.nFeat = nFeat
        self.nClasses = nClasses
        self.resBlock = resBlock
        self.inplanes = inplanes
 
        self._make_head()
 
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(nStacks):
            hg.append(Hourglass(4, nFeat, nModules, resBlock))
            res.append(self._make_residual(nModules))
            fc.append(self._make_fc(nFeat, nFeat))
            score.append(nn.Conv2d(nFeat, nClasses, 1))
            if i < (nStacks - 1):
                fc_.append(nn.Conv2d(nFeat, nFeat, 1))
                score_.append(nn.Conv2d(nClasses, nFeat, 1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)
        self.hg2 = Hourglass2(4, nFeat, nModules, resBlock)
        self.up2t = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4t = nn.Upsample(scale_factor=4, mode='nearest')
        self.up8t = nn.Upsample(scale_factor=8, mode='nearest')
        self.fconv = nn.Conv2d(self.inplanes, nFeat)

    def _make_head(self):
        self.conv1 = nn.Conv2d(self.inplanes, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = self.resBlock(128, 128)
        self.res3 = self.resBlock(128, self.nFeat)
 
    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])
 
    def _make_fc(self, inplanes, outplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True))
 
    def forward(self, x):
        # head
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
 
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.res3(x)
 
        out = []
        last_out = []
        for i in range(self.nStacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < (self.nStacks - 1):
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        
        x，low1,low2,low3 = self.hg2(x)

        out2 = self.fconv(x+low1+low2+low3)

        return out,out2



netHg = HourglassNet(nStacks, nModules, nFeat, nClasses)



#########################################################
criterion = torch.nn.MSELoss(size_average=True)
# target_var: single image keypoint heatmap. N by W by H
# N is the size of key_pt
# out: batch image keypoint heatmap: B by N by W by H
# B is batch size
for img_id, structure in zip(img_list, structure_list):
	loss_single_pt = criterion(out[img_id,:,:,:], target_var)
	for s in structure:
		loss_single_pt += criterion(out[img_id,s[0],:,:]+out[img_id,s[1],:,:], target_var[s[0],:,:]+target_var[s[1],:,:])

	LSA += loss_single_pt+alpha*loss_single_pt

for img_id in img_list:
	loss_single_pt_MS = criterion(out2[img_id,:,:,:], target_var)
	LMS += loss_single_pt_MS









