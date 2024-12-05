function [ yhat_all, LgradB_all, LgradT_all, t_min, t_max ] = modelKA_basisC( x, xmin, xmax, ymin, ymax, fnB, fnT )

fprintf("Dims of x received by modelKA_basisC: ")
size(x)


N = size(x,1)
m = size(x,2);
n = size(fnB,1);
q = size(fnT,1);
p = size(fnT,2)

fprintf("N, m, n, q, p = %d %d %d %d %d \n", N, m, n, q, p)

tmin = ymin;
tmax = ymax;

%. proj. matrices
Cpq = kron(eye(p),ones(q,1));
Cpnm = kron(eye(p),ones(1,n*m));

Mn = splineMatrix(n);
Mq = splineMatrix(q);

fnB_r = reshape(fnB,n*m,[]);

fprintf("New dims of fnB: ")
size(fnB_r)

fnT_r = fnT(:);
fprintf("Dims of fnT_r: ")
size(fnT_r)

%. calc. bottom
xr = reshape(x.',1,N*m);
[ phi, dphi, ddphi ] = basisFunc_spline( xr, xmin, xmax, n, Mn );
phi_r = reshape(phi,n*m,N);
t = phi_r.' * fnB_r;

fprintf("Dims of t: ")
size(t)


%. calc. top
tr = reshape(t.',1,N*p);
[ psi, dpsi, ddpsi, dddpsi ] = basisFunc_spline( tr, tmin, tmax, q, Mq );
psi_r = reshape(psi,q*p,N);
yhat_all = psi_r.' * fnT_r;

%. deriv.
dpsi_r = reshape(dpsi,q*p,N);
top = dpsi_r.' * diag(fnT_r.') * Cpq;
phi_re = repmat(phi_r.',1,p);
LgradB_all = ( top * Cpnm ) .* phi_re;
LgradT_all = psi_r.';

t_min = min(t);
t_max = max(t);

end
