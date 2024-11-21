function [ fnB, fnT ] = buildKA_init( m, n, q, p, ymin, ymax )

%. limits
tmin = ymin;
tmax = ymax;

%. init. operators
amplB = (tmax-tmin)/m;
amplT = (ymax-ymin)/p;
fnB = tmin/m + rand( n, p*m ) * amplB;
fnT = ymin/p + rand( q, p ) * amplT;

end
