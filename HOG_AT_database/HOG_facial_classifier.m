%% Load Image information from ATT Face database direcotry
faceDatabase = imageDatastore('archive', ...
    'IncludeSubfolders', true, ... % Allow searching in subfolders
    'LabelSource', 'foldernames') % Assign labels on images based on the folders

%% Display Monatage of s1

labels = faceDatabase.Labels; % Get all labels

figure(1);
image_folder = find(labels == 's1'); 
imshow(imread(faceDatabase.Files{image_folder(2)})); %dispay 2nd image from s1
title('One image of single face');

figure(2);
montage(faceDatabase.Files(labels=='s1')); %display all images from s1
title('All images of single face');
%% Spliting database into training and test sets
[trainingSet, testSet] = splitEachLabel(faceDatabase,0.8,0.2);
%% Extracting features from all of the people

trainingFeatures = zeros(numel(trainingSet.Files), 4680); % Preallocate (HOG default size)
trainingLabels = cell(numel(trainingSet.Files), 1);      % Preallocate labels
personIndex = cell(numel(trainingSet.Labels, 1), 1);

featureCount = 1;
for i = 1:numel(unique(trainingSet.Labels)) % loops through subfolders
    subds = subset(trainingSet, trainingSet.Labels == unique(trainingSet.Labels(featureCount)));
    for j = 1:numel(subds.Files) %loops through images
        img = read(subds);
        trainingFeatures(featureCount, :) = extractHOGFeatures(img);
        trainingLabels{featureCount} = char(unique(trainingSet.Labels(featureCount)));
        featureCount = featureCount + 1;
    end
    personIndex{i} = char(unique(trainingSet.Labels(i)));
end
%% creating a 40 class clasifier
faceClassifier = fitcecoc(trainingFeatures, categorical(trainingLabels))
% syntax fit+classification/regression+method(ecoc)
%% Test Images from Test set
uniqueTestLabels = unique(testSet.Labels); % Get all unique person labels

person_lab = 1; % Test the first person in test set

des_sub=subset(testSet, testSet.Labels == uniqueTestLabels(person_lab));
desiredImage = read(des_sub);% Get test image
desiredFeatures = extractHOGFeatures(desiredImage); % extract features from the image
predLabel = predict(faceClassifier, desiredFeatures); %predict the match based on the clasifier

% Display results
figure;
subplot(1,2,1); imshow(desiredImage); title(['Test face: ' char(uniqueTestLabels(person_lab))]);
subplot(1,2,2); imshow(read(subset(trainingSet, trainingSet.Labels == predLabel))); 
title(['Predicted: ' char(predLabel)]);
%% Test for 5 different people
figure;
peopleRange = 5:10;

for rangeIdx = 1:min(5, numel(peopleRange))  % Max 5 people (for 2x5 grid)
    personIdx = peopleRange(rangeIdx);
    % Get all test images for this person
    currentLabel = uniqueTestLabels(personIdx);
    personTestSet = subset(testSet, testSet.Labels == currentLabel);
    reset(personTestSet);
    
    % Test first 2 images per person
    for imgIdx = 1:min(2, numel(personTestSet.Files))
        % Read and process image
        queryImage = read(personTestSet);
        queryFeatures = extractHOGFeatures(queryImage);
        predLabel = predict(faceClassifier, queryFeatures);
        
        % Find matching training image
        trainMatch = subset(trainingSet, trainingSet.Labels == predLabel);
        reset(trainMatch);
        matchedImage = read(trainMatch);
        
        % Display results (2x5 grid)
        subplot(2, 5, (rangeIdx-1)*2 + imgIdx);
        imshowpair(queryImage, matchedImage, 'montage');
        title(sprintf('Test: %s\nPred: %s', char(currentLabel), char(predLabel)));
    end
end