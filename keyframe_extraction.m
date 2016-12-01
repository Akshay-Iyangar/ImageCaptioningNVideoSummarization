   vid = 'news.mp4';
   

   readerobj = VideoReader(vid);
   X= zeros(1,readerobj.NumberOfFrames);
   
   for k=1:  readerobj.NumberOfFrames
           I=read(readerobj,k);
           if(k~= readerobj.NumberOfFrames)
                 J=read(readerobj,k+1);
                 sss= sum(sum(sum(abs(I-J))));
                 %sss=imabsdiff(I,J);
                 X(k)=sss;
                 %frame_mean(k)= mean2(X(k,:,:,:));
           end
   end
   mean=mean2(X);
   std=std2(X);
   threshold=std+2*mean;
   i=1;
   for k=1: readerobj.NumberOfFrames
       I =  read(readerobj,k);
       if(k~=readerobj.NumberOfFrames)
        J=   read(readerobj,k+1);
        sss=sum(sum(sum(abs(I-J))));
        if(sss > threshold)
            sprintf('inside loop')
            imwrite(J,strcat('/Users/akshayiyangar/Documents/MATLAB/Caption/',sprintf('%05d',i),'.jpg'));
            i = i + 1;
        end
       end
   end
   