list1 = [1,3,4,5,6,8,9]
list2 = [1,8,9]
list3 = list()
i = 0
j = 0
while (i < len(list1)) & (j < len(list2)):
    if list1[i] == list2[j]:
        list3.append(list1[i])
        i += 1
        j += 1
    elif list1[i] < list2[j]:
        i += 1
    elif list1[i] > list2[j]:
        j += 1
print(list3)