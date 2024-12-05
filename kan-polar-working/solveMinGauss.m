function [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = solveMinGauss( x, y, lab, identID, verifID, alp, lam, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 )

% size(x)

m = size(x, 1) * size(x, 2);

fnB = fnB0;
fnT = fnT0;
n = size(fnB,1);
q = size(fnT,1);
p = size(fnT,2);

RMSE = zeros(Nrun,1);
t_min_all = zeros(Nrun,p);
t_max_all = zeros(Nrun,p);

% size(x, 3)

for jj=1:Nrun
    
    indI = 1:identID;
    indV = verifID:size(x, 3);


    %. training
    for i = 1:size(x,3) 
        
        % fprintf("Current indI: %d \n", indI)

        x_step = x(:,:, i);

        fprintf("Dims of current x_step: ")
        size(x_step)

        fprintf("Dims of matrix passed to modelKA_basisC: ")
        size(x_step)

        [ yhat_all, LgradB_all, LgradT_all ] = modelKA_basisC( x_step, xmin, xmax, ymin, ymax, fnB, fnT );
        L_all = yhat_all - y(indI,:);
    
        F = L_all;
        J = [ LgradB_all LgradT_all ];
        A = J.' * J;
        b = J.' * F;
        Ar = A + lam*eye(n*p*m+q*p);

        dlt = -Ar\b;
        dltB = reshape(dlt(1:(n*p*m)),n,[]);
        dltT = reshape(dlt((n*p*m+1):end),q,[]);

        fnB = fnB + alp*dltB;
        fnT = fnT + alp*dltT;
    end 

    %. validation
    for i = 1:size(x,3)
        [ yhat_all, dum1, dum2, t_min, t_max ] = modelKA_basisC( x(:,:, indV), xmin, xmax, ymin, ymax, fnB, fnT );
        err_all = abs( yhat_all - y(indV,:) );
        RMSE(jj) = sqrt( mean( err_all.^2 ) )/(ymax-ymin);
        t_min_all(jj,:) = t_min;
        t_max_all(jj,:) = t_max;
    end
    
    printProgr = 1;
    if ( printProgr == 1 )
        if ( jj > 1 )
            fprintf( repmat( '\b', 1, 34 ) );
        end
        fprintf( '  pass %04.0f out of %04.0f completed\n', jj, Nrun );
    end
end

end
