clear all;
close all;
clc;

%% k-Nearest Neighbor Algoritm

%% Vars

% A : connectivity matrix
% F_set : input vectors
% T_set : desired output vectors
% G_set : actual output vectors

% dim : dimensions
% count : amount of associations

% cos : mean cosine angle between t and g 

%% Generating stimuli

% Parameters
dim = 100;
count = 80;

% Create simuli
[A,F_set,T_set] = generate_stim(dim,count);

%% Training & Testing

disp('Start training!');
A = train(A,F_set,T_set);
disp('Done training!');

cos = test(A,F_set,T_set);
disp('Performance');
disp(cos);

%% a.) Oscillations

clear all;
close all;
clc;

% i.)

% Parameters
dim = 100;
count = 80;
k=0.1;

% Create stimuli
[A,F_set,T_set] = generate_stim(dim,count);

% Training
disp('Start training!');
A = ai_train(A,F_set,T_set,k);
disp('Done training!');

% Testing
cos = test(A,F_set,T_set);
disp('Performance');
disp(cos);

% When the associations was 20, the cos angle was at 0.95
% When the associations was 40, the cos angle was at 0.88
% When the associations was 60, the cos angle was at 0.82
% When the associations was 80, the cos angle was at 0.78

%% ii.)

% Parameters
dim = 100;
count = 80;

% Create stimuli
[A,F_set,T_set] = generate_stim(dim,count);

% Training
disp('Start training!');
A = aii_train(A,F_set,T_set);
disp('Done training!');

% Testing
cos = test(A,F_set,T_set);
disp('Performance');
disp(cos);

% When the associations was 20, the cos angle was at 0.95
% When the associations was 40, the cos angle was at 0.95
% When the associations was 60, the cos angle was at 0.95
% When the associations was 80, the cos angle was at 0.93

% I found that the second method was better because the cosine angle kept
% staying around 0.95 despite increasing the amount of associations. 
% When using a fixed number, I had to increase the amount of iterations 
% by a lot to acheive the results of the second method. For instance,
% when k=0.1, I had to present each vector 150 times instead of 25 to get
% the cos to equal 1.0.

% Overall, the errors keep going down. However, when I print out the
% errors, I notice that there is a good amount of oscillations in the
% second method. When there is a fixed k, there are no oscillations

%% b.) Convergence

clear all;
close all;
clc;

% Parameters
dim = 100;
count = 80;

% Create stimuli
[A,F_set,T_set] = generate_stim(dim,count);

% Training
disp('Start training!');
A = b_train(A,F_set,T_set);
disp('Done training!');

% Testing
cos = test(A,F_set,T_set);
disp('Performance');
disp(cos);

% Convergence occured at varying iterations depending on the count.
% When the count was 20, it took 600-1000 iterations
% When the count was 40, it took 4k-5k iterations
% When the count was 60, it took 14k-18k iterations
% When the count was 80, it took 50k-65k iterations

%% c.) Deterioration

clear all;
close all;
clc;

% Parameters
dim = 200;
count = 1200;

% Create stimuli
[A,F_set,T_set] = generate_stim(dim,count);

% Training
disp('Start training!');
A = train(A,F_set,T_set);
disp('Done training!');

% Testing
cos = test(A,F_set,T_set);
disp('Performance');
disp(cos);

% Generate new, random input vector called H
h = rand(dim,1);
h = h - mean(h);
h = h/norm(h); 
% Compute predicted output
h_o = A*h;

% Select a random input vector
rand = randi([1 count],1);
t = T_set(:,rand);
f = F_set(:,rand);
% Compute predicted output
f_o = A*f;

% Compute the cosine angle
disp('Chance:');
disp(dot(t,h_o));
disp('Our NN:');
disp(dot(t,f_o));

% I found that after 200 associations, the neural network's performance 
% started deteriorating fast. Before 200 associations, the cosine angle
% didn't go below 0.9. The neural network also started doing only just as
% well as chance when there were about 1500 associations.

%% d.) Sequential learning

clear all;
close all;
clc;

%% Forwards

% Parameters
dim = 100;
count = 60;

% Create stimuli
[A,F_set,T_set] = generate_stim(dim,count);

% Training
disp('Start training!');
A = di_train(A,F_set,T_set);
disp('Done training!');

% Testing
cos = test(A,F_set,T_set);
disp('Performance');
disp(cos);

%% Backwards

% Parameters
dim = 100;
count = 80;

% Create stimuli
[A,F_set,T_set] = generate_stim(dim,count);

% Training
disp('Start training!');
A = dii_train(A,F_set,T_set);
disp('Done training!');

% Testing
cos = test(A,F_set,T_set);
disp('Performance');
disp(cos);

% They both performed very similarly in terms of their cosine angle.
% When you present the vectors in sequence over and over again, you
% could potentially ingrain the sequence into the A matrix.

% Because the error is calculated using the A matrix, and then uses that
% error to change the A matrix, the current error is dependent on the
% previous error. So when you keep providing the same sequence over
% and over again, the matrix could become dependent on inputs being
% sequential as well.

% There didn't seem to be any difference between learning forwards
% versus learning backwards. But the cosine angle was higher in the
% sequential method (~0.98) versus the random method (~0.95) when the
% amount of associations to be learned were 80. It was even better
% when 60 had to be learned.


%% Generate connectivity matrix, i/o vectors, and predicted output
function [A,F_set,G_set] = generate_stim(dim,count)

    % Allocate memory for set of i/o vectors
    F_set=zeros(dim,count);
    G_set=zeros(dim,count);
    
    % Allocate memory for set of A matricies
    A_set=zeros(dim,dim,count);
    
    % Generate random i/o vectors and connectivity matrix
    for index=1:count
        G = rand(dim,1);
        G = G - mean(G); % subtract mean
        G = G/norm(G); % normalize
        F = rand(dim,1);
        F = F - mean(F); % subtract mean
        F = F/norm(F); % normalize
        A = G*F.'; % generate a connectivity matrix
        
        % remember the sets of vectors
        F_set(:,index) = F;
        G_set(:,index) = G;
        A_set(:,:,index) = A;
    end
    A = sum(A_set,3); % add up the A matricies
end

%% Training function
function [A] = train(A,F_set,T_set)

    % Dimensions
    dim = size(F_set);
    
    % Store the error^2 vectors to track error
    e2 = [];
    
    % How many times to present each vector
    iterations = 25;
    
    for j=1:iterations
        for i=1:dim(2)
            % Choose a random i/o pair
            rand = randi([1 dim(2)],1);
            f = F_set(:,rand);
            t = T_set(:,rand);
            % Compute k (learning constant)
            k = 1/(f.'*f);
            % Actual output vector g
            g = A*f;
            % Compute error vector
            error = (t-g);
            % Delta A 
            delta = k*error*f';
            % Add both A and delta A
            A=A+delta;
            % Save length of error vector in list
            e2(end+1) = norm(error);
            disp(mean(e2));
        end
    end
end

%% Testing function
function [cos] = test(A,F_set,T_set);
    dim=size(F_set);
    cos=[];
    for i=1:dim(2)
        t=T_set(:,i);
        f=F_set(:,i);
        g=A*f;
        cos(end+1)=dot(t,g);
    end
    cos=mean(cos);
end

%% Functions for specific questions

% a.) i.)
function [A] = ai_train(A,F_set,T_set,k)
    % Dimensions
    dim = size(F_set);
    % Store the error^2 vectors to track error
    e2 = [];
    % How many times to present each vector
    iterations = 25;
    for j=1:iterations
        for i=1:dim(2)
            % Choose a random i/o pair
            rand = randi([1 dim(2)],1);
            f = F_set(:,rand);
            t = T_set(:,rand);
            % Actual output vector g
            g = A*f;
            % Compute error vector
            error = (t-g);
            % Delta A 
            delta = k*error*f';
            % Add both A and delta A
            A=A+delta;
            % Save length of error vector in list
            e2(end+1) = norm(error.^2);
            disp(mean(e2));
        end
    end
end

% a.) ii.)
function [A] = aii_train(A,F_set,T_set)
    % Dimensions
    dim = size(F_set);
    % Store the error^2 vectors to track error
    e2 = [];
    % How many times to present each vector
    iterations = 25;
    index=0;
    for j=1:iterations
        for i=1:dim(2)
            % Increment index
            index = index + 1;
            % Choose a random i/o pair
            rand = randi([1 dim(2)],1);
            f = F_set(:,rand);
            t = T_set(:,rand);
            % Compute k (learning constant)
            k = ((1/(f.'*f))-0.001)/index;
            % Actual output vector g
            g = A*f;
            % Compute error vector
            error = (t-g);
            % Delta A 
            delta = k*error*f';
            % Add both A and delta A
            A=A+delta;
            % Save length of error vector in list
            e2(end+1) = norm(error.^2);
            disp(mean(e2));
        end
    end
end

% b.)
function [A] = b_train(A,F_set,T_set)

    % Dimensions
    dim = size(F_set);
    
    % Store the error^2 vectors to track error
    e2 = [];
    
    % How many times to present each vector
    iterations = 10000;
    
    index=0;
    
    for j=1:iterations
        for i=1:dim(2)
            % Choose a random i/o pair
            rand = randi([1 dim(2)],1);
            f = F_set(:,rand);
            t = T_set(:,rand);
            % Compute k (learning constant)
            k = 1/(f.'*f);
            % Actual output vector g
            g = A*f;
            % Compute error vector
            error = (t-g);
            % Delta A 
            delta = k*error*f';
            % Add both A and delta A
            A=A+delta;
            % Save length of error vector in list
            e2(end+1) = norm(error.^2);
            disp(mean(e2));
            index = index+1;
            if mean(e2) <= 0.001
                disp(index);
                break
            end
        end
        if mean(e2) <= 0.001
            break
        end
    end
end

% d.) i.)
function [A] = di_train(A,F_set,T_set)

    % Dimensions
    dim = size(F_set);
    
    % Store the error^2 vectors to track error
    e2 = [];
    
    % How many times to present each vector
    iterations = 25;
    
    for j=1:iterations
        for i=1:dim(2)
            f = F_set(:,i);
            t = T_set(:,i);
            % Compute k (learning constant)
            k = 1/(f.'*f);
            % Actual output vector g
            g = A*f;
            % Compute error vector
            error = (t-g);
            % Delta A 
            delta = k*error*f';
            % Add both A and delta A
            A=A+delta;
            % Save length of error vector in list
            e2(end+1) = norm(error.^2);
            disp(mean(e2));
        end
    end
end

% d.) ii.)
function [A] = dii_train(A,F_set,T_set)

    % Dimensions
    dim = size(F_set);
    
    % Store the error^2 vectors to track error
    e2 = [];
    
    % How many times to present each vector
    iterations = 25;
    
    for j=1:iterations
        for i=1:dim(2)
            f = F_set(:,dim(2)+1-i);
            t = T_set(:,dim(2)+1-i);
            % Compute k (learning constant)
            k = 1/(f.'*f);
            % Actual output vector g
            g = A*f;
            % Compute error vector
            error = (t-g);
            % Delta A 
            delta = k*error*f';
            % Add both A and delta A
            A=A+delta;
            % Save length of error vector in list
            e2(end+1) = norm(error.^2);
            disp(mean(e2));
        end
    end
end

%% This took about 7 hours to complete