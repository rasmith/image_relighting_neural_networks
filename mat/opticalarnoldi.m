% OPTICALARNOLDI.M   Script to test non-symmetric algorithm 2 (optical
% Arnoldi for rank-K transport acquisition) of "Optical Computing for Fast
% Light Transport Analysis".  See:
%     http://www.dgp.toronto.edu/~motoole/opticalcomputing.html
%
% Author: Matthew O'Toole (motoole@dgp.toronto.edu)
% Published: July 2013
% Version: 1.0
%

USE_SYNTHETIC_LTM = 1;     % Use a synthetic light transport matrix
NUM_ITERATIONS = 1024;     % Number of Arnoldi iterations
NUM_ITERATIONS_STEP = 1;   % Only reconstruct matrix every 
                           % NUM_ITERATIONS_STEP iterations

                           
                           
%% Load Light Transport Matrix (LTM) into memory
fprintf('Load light transport matrix (LTM) into memory... ');

if (USE_SYNTHETIC_LTM)
    SIZE = 128;
    RANK = 64;
    
    % Generate a random non-symmetric low-rank light transport matrix.
    LTM = randn(SIZE,RANK)*randn(RANK,SIZE);
    
    % Make sure the light transport matrix is not exactly low-rank to avoid
    % numerical issues with the orthogonalization step.
    LTM = LTM + 1e-3.*randn(SIZE);
else
    % Load measured light transport matrix dataset.
    load 'Green.mat'
    
    % The supplied light transport matrices have size [464x696] x [32x32], 
    % representing a 464x696 photo of the scene lit by a 32x32 light 
    % pattern.
    % To visualize the light transport matrix, reshape any linear 
    % combination of columns into an image of size 464x696.
    % i.e. imagesc(reshape(sum(LTM,2),sizeIm));
    
    % Because comparing the reconstructed LTM to the ground-truth LTM takes
    % some time, only do this once for every 32 photos.
    NUM_ITERATIONS_STEP = 32;
end

fprintf('done.\n');

% Get size of matrix
[M,N] = size(LTM);

% Number of Arnoldi iterations should not exceed matrix dimensions.
NUM_ITERATIONS = min(NUM_ITERATIONS,min(M,N));



%% Apply the non-symmetric optical Arnoldi algorithm to the LTM

% Start with an initial vector of all ones.
right_Arnoldi_vectors = ones(N,1);
right_Arnoldi_vectors = right_Arnoldi_vectors./norm(right_Arnoldi_vectors);
left_Arnoldi_vectors  = zeros(M,0);

fprintf('Measure LTM using Arnoldi method: ');
num_chars = 0;

for k = 1:NUM_ITERATIONS
    fprintf(repmat('\b',[1 num_chars]));
    num_chars = fprintf('%d / %d',k,NUM_ITERATIONS);
    
    % Simulate project-and-capture operation using 
    % right_Arnoldi_vector(:,k) as the illumination pattern.
    tmp = LTM*right_Arnoldi_vectors(:,k);    
    left_Arnoldi_vectors(:,k) = tmp;
    
    % Simulate project-and-capture operation using 
    % left_Arnoldi_vector(:,k) as the illumination pattern.
    tmp = LTM'*left_Arnoldi_vectors(:,k);
    
    % Make vector orthonormal to current set of right_Arnoldi_vectors.
    % To project a vector x into an orthogonal subspace of P, evaluate
    %     x - P*inv(P'*P)*P'*x
    tmp = tmp - right_Arnoldi_vectors*...
        ((right_Arnoldi_vectors'*right_Arnoldi_vectors)\...
        (right_Arnoldi_vectors'*tmp));
    tmp = tmp./norm(tmp);

    right_Arnoldi_vectors(:,k+1) = tmp;
end

fprintf('\n');



%% Reconstruct LTM using Arnoldi vectors

Arnoldi_residual_error = [];

fprintf('Reconstruct LTMs using Arnoldi method: ');
num_chars = 0;

for k = NUM_ITERATIONS_STEP:NUM_ITERATIONS_STEP:NUM_ITERATIONS
    fprintf(repmat('\b',[1 num_chars]));
    num_chars = fprintf('%d / %d',k,NUM_ITERATIONS);
    
    % Compute rank-k approximation of light transport matrix using Arnoldi
    % vectors.
    LTM_approx = left_Arnoldi_vectors(:,1:k)*right_Arnoldi_vectors(:,1:k)';
    
    % Compute the relative Frobenius norm of the residual.
    Arnoldi_residual_error(end+1) = norm(LTM(:)-LTM_approx(:))./...
        norm(LTM(:));
end

fprintf('\n');

clear left_Arnoldi_vectors right_Arnoldi_vectors LTM_approx



%% Apply SVD to the LTM

fprintf('Compute SVD of LTM... ');

% MATLAB's SVD applied directly to large matrices is *very* slow.  This is 
% an issue when computing the SVD of the measured LTM, which has size 
% 322944x1024.  Instead, the SVD of LTM'*LTM, a 1024x1024 matrix, produces
% the right singular vectors of LTM much more efficiently.  Last, multiply
% LTM with the right singular vectors to produce the left singular vectors 
% of LTM (scaled by the singular values).

% Compute right singular vectors
tmp = LTM'*LTM;
[U,S,right_singular_vectors] = svd(tmp);

% Compute left singular vectors
left_singular_vectors = LTM*right_singular_vectors;

fprintf('done.\n')



%% Reconstruct LTM using singular vectors

svd_residual_error = [];

fprintf('Reconstruct LTMs using SVD method: ');
num_chars = 0;

for k = NUM_ITERATIONS_STEP:NUM_ITERATIONS_STEP:NUM_ITERATIONS
    fprintf(repmat('\b',[1 num_chars]));
    num_chars = fprintf('%d / %d',k,NUM_ITERATIONS);
    
    % Compute rank-k approximation of light transport matrix using singular
    % vectors.
    LTM_approx = left_singular_vectors(:,1:k)*...
        right_singular_vectors(:,1:k)';
    
    % Compute the relative Frobenius norm of the residual
    svd_residual_error(end+1) = norm(LTM(:)-LTM_approx(:))./norm(LTM(:));
end

fprintf('\n');

clear left_singular_vectors right_singular_vectors LTM_approx


%% Plot results

figure(1); clf;
hold on;
plot(NUM_ITERATIONS_STEP:NUM_ITERATIONS_STEP:NUM_ITERATIONS,...
     Arnoldi_residual_error,'r',...
     NUM_ITERATIONS_STEP:NUM_ITERATIONS_STEP:NUM_ITERATIONS,...
     svd_residual_error,'k');
legend('Arnoldi','Singular value decomposition');
hold off;

axis([1 NUM_ITERATIONS 0 1]);
xlabel('Number of iterations');
ylabel('Relative residual error');
