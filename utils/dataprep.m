p = p';
t = t(1:3,:)';
meshfile = [length(p),length(t),0];

dlmwrite('mesh_file.dat',meshfile,'Delimiter','\t','precision','%.0f')
dlmwrite('mesh_file.dat',p,'-append','Delimiter','\t','precision','%.8f')
dlmwrite('mesh_file.dat',t,'-append','Delimiter','\t','precision','%.0f')