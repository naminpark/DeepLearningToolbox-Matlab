function Data = GetFieldByIndex(S, n)
    D    = struct2cell(S);
    Data = D{n};   % Care for struct arrays!

end