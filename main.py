import model
import tools

if __name__ == '__main__':

    args = tools.getArgs()

    if args.command == 'fit':
        pass
    elif args.command == 'predict':
        pass
    elif args.command == 'score':
        pass

    #funciton choose:
        #fit
            #X_dataset
            #Y_dataset
            #output
        #predict
            #model_file
            #X_dataset
            #output
        #score
            #X_dataset
            #Y_dataset
            #confusionMatrixOut
    pass