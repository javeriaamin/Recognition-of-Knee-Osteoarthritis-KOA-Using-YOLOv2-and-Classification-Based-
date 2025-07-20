inputsize=[224 224]
I = imread('grade4 (9).png');
I = imresize(I,inputsize(1:2));

[bboxes,scores] = detect(detectorTinyYolo2,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

[bboxes,scores,labels] = detect(detectorTinyYolo2,I)
if ~isempty(bboxes)    BBarea = bboxes(:,1).*bboxes(:,2)
    BBarea = bboxes(:,1).*bboxes(:,2)

    
    detectedI = insertObjectAnnotation(I,'rectangle',bboxes,cellstr(labels));
end
figure
imshow(detectedI)
