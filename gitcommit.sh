echo 开始添加变更：git add .
git add .
echo;

echo 查看暂存区状态：git status 
git status
echo;

read -r -p "是否将将暂存区内容添加到本地仓库 [y/n]" input
 
case $input in
    [yY][eE][sS]|[yY])
        echo 暂存区内容添加到本地仓库中：git commit 
        git commit -m $1
        echo;
        
        read -r -p "是否将变更情况提交到远程自己分支 [y/n]" input
 
        case $input in
            [yY][eE][sS]|[yY])
                echo 提交到远程自己分支中：git push -u origin master
                git push -u origin master
                echo;
        exit 1
        ;;

            [nN][oO]|[nN])
                echo "中断提交"
                exit 1
                ;;

            *)
            echo "输入错误，请重新输入"
            ;;
        esac
        
        exit 1
        ;;

    [nN][oO]|[nN])
        echo "中断提交"
        exit 1
        ;;

    *)
    echo "输入错误，请重新输入"
    ;;
esac

echo 执行完毕！
echo;
 
pause
